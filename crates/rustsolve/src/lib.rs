//! Sparse direct solves for finite-element systems: LL^T, LDL^T and LU on
//! [faer](https://codeberg.org/sarah-quinones/faer), behind a small
//! analyze → factorize → solve interface with explicit errors.
//!
//! Written to replace the CHOLMOD/UMFPACK backend of pyGIMLi's `LinSolver`
//! (see `docs/design.md`). The rules the interface keeps:
//!
//! - nothing is perturbed: faer's dynamic regularization stays off and is not
//!   exposed;
//! - no stale factor: a failed factorization leaves the solver with no factor,
//!   so a later solve fails with [`Error::NotFactorized`];
//! - an analysis is reused only for an identical pattern, compared entry by
//!   entry (never by size, count or hash alone);
//! - no fallback inside the library: the caller chooses the factorization.

mod cache;

pub use cache::clear_shared_analyses;

use std::sync::Arc;

use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::cholesky::ldlt::factor::LdltRegularization;
use faer::linalg::cholesky::llt::factor::LltRegularization;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, LdltRef, LltRef, SymbolicCholesky,
    SymbolicCholeskyRaw, SymmetricOrdering,
};
use faer::sparse::linalg::lu::{factorize_symbolic_lu, LuRef, LuSymbolicParams, NumericLu, SymbolicLu};
use faer::sparse::linalg::{LuError, SupernodalThreshold};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Conj, MatMut, Par, Side};

/// Which factorization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// Cholesky LL^T: symmetric positive definite matrices.
    Llt,
    /// LDL^T without pivoting: symmetric matrices with nonzero pivots.
    Ldlt,
    /// LU with partial pivoting: general square matrices.
    Lu,
}

/// Fill-reducing ordering of the symmetric factorizations (LU always uses
/// COLAMD).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ordering {
    /// Approximate minimum degree (faer's).
    Amd,
    /// Multilevel nested dissection (feral-metis); needs the `nd` feature.
    NestedDissection,
}

/// Simplicial or supernodal kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Nodal {
    /// faer chooses from the operation count.
    Auto,
    Simplicial,
    Supernodal,
}

/// Solver settings.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub kind: Kind,
    pub ordering: Ordering,
    pub nodal: Nodal,
    /// Symmetric kinds: refuse a matrix with an off-diagonal entry `v` whose
    /// mirror `m` gives |v - m| > `symmetry_tol` * max(|v|, |m|). A mirror that
    /// is not stored counts as zero, so a nonzero entry without one is refused
    /// (for `symmetry_tol` < 1). Without the check, the factorization reads
    /// only the lower triangle of the column view (the upper triangle of
    /// compressed rows); every stored value is still checked to be finite.
    pub check_symmetry: bool,
    pub symmetry_tol: f64,
    /// The arrays are compressed rows (CSR) rather than compressed columns.
    /// The solver always solves with the matrix the arrays describe.
    pub compressed_rows: bool,
    /// LL^T / LDL^T: refuse a factor whose smallest |pivot| / largest |pivot|
    /// is below this. `None` = n * eps, `Some(0.0)` = no check.
    pub singular_pivot_ratio: Option<f64>,
    /// LU: refuse a factor whose reciprocal condition estimate is below this
    /// (a few extra solves). `None` = eps, `Some(0.0)` = no check.
    pub min_rcond: Option<f64>,
    /// Threads for the numeric factorization: 1 = sequential, 0 = all cores.
    pub factor_threads: usize,
    /// Threads for the triangular solves: 1 = sequential, 0 = all cores.
    pub solve_threads: usize,
    /// Reuse the symbolic analysis of an identical pattern made by another
    /// solver of the same process (pyGIMLi builds a new solver for every
    /// matrix of a forward run). Numeric factors are never shared.
    pub share_analysis: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            kind: Kind::Llt,
            ordering: Ordering::Amd,
            nodal: Nodal::Auto,
            check_symmetry: true,
            symmetry_tol: 1e-12,
            compressed_rows: false,
            singular_pivot_ratio: None,
            min_rcond: None,
            factor_threads: 1,
            solve_threads: 1,
            share_analysis: false,
        }
    }
}

/// Every way a call can fail. `code()` is the C ABI status.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    InvalidInput(String),
    /// a symmetric kind with a column that has no diagonal entry
    StructurallySingular { column: usize },
    NotSymmetric { row: usize, col: usize },
    NonFiniteInput { position: usize },
    PatternChanged,
    NotAnalyzed,
    NotFactorized,
    /// pivot at elimination `step` (0-based) is <= 0; `column` is the original
    /// column eliminated there
    NotPositiveDefinite { step: usize, column: usize },
    ZeroPivot { step: usize, column: usize },
    /// smallest |pivot| / largest |pivot| (LL^T, LDL^T) or the reciprocal
    /// condition estimate (LU) below the threshold
    NumericallySingular { step: Option<usize>, column: Option<usize>, value: f64 },
    /// LU found no pivot candidate at `step`
    Singular { step: usize },
    NonFiniteSolution,
    OutOfMemory,
    Unsupported(String),
    Internal(String),
}

impl Error {
    pub fn code(&self) -> i32 {
        match self {
            Error::InvalidInput(_) => 1,
            Error::StructurallySingular { .. } => 2,
            Error::NotSymmetric { .. } => 3,
            Error::NonFiniteInput { .. } => 4,
            Error::PatternChanged => 5,
            Error::NotAnalyzed => 6,
            Error::NotFactorized => 7,
            Error::NotPositiveDefinite { .. } => 8,
            Error::ZeroPivot { .. } => 9,
            Error::NumericallySingular { .. } => 10,
            Error::Singular { .. } => 11,
            Error::NonFiniteSolution => 12,
            Error::OutOfMemory => 13,
            Error::Internal(_) => 14,
            Error::Unsupported(_) => 16,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Error::InvalidInput(_) => "INVALID_INPUT",
            Error::StructurallySingular { .. } => "STRUCTURALLY_SINGULAR",
            Error::NotSymmetric { .. } => "NOT_SYMMETRIC",
            Error::NonFiniteInput { .. } => "NON_FINITE_INPUT",
            Error::PatternChanged => "PATTERN_CHANGED",
            Error::NotAnalyzed => "NOT_ANALYZED",
            Error::NotFactorized => "NOT_FACTORIZED",
            Error::NotPositiveDefinite { .. } => "NOT_POSITIVE_DEFINITE",
            Error::ZeroPivot { .. } => "ZERO_PIVOT",
            Error::NumericallySingular { .. } => "NUMERICALLY_SINGULAR",
            Error::Singular { .. } => "SINGULAR",
            Error::NonFiniteSolution => "NON_FINITE_SOLUTION",
            Error::OutOfMemory => "OUT_OF_MEMORY",
            Error::Internal(_) => "INTERNAL",
            Error::Unsupported(_) => "UNSUPPORTED",
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidInput(s) => write!(f, "invalid input: {s}"),
            Error::StructurallySingular { column } => write!(f, "column {column} has no diagonal entry"),
            Error::NotSymmetric { row, col } => write!(f, "entries ({row}, {col}) and ({col}, {row}) differ"),
            Error::NonFiniteInput { position } => write!(f, "value {position} is not finite"),
            Error::PatternChanged => write!(f, "the sparsity pattern differs from the analyzed one"),
            Error::NotAnalyzed => write!(f, "no analysis: call analyze or update first"),
            Error::NotFactorized => write!(f, "no valid factorization"),
            Error::NotPositiveDefinite { step, column } => {
                write!(f, "matrix not positive definite: pivot {step} (column {column}) <= 0")
            }
            Error::ZeroPivot { step, column } => write!(f, "zero or non-finite pivot {step} (column {column})"),
            Error::NumericallySingular { step, column, value } => match (step, column) {
                (Some(s), Some(c)) => write!(f, "numerically singular: pivot ratio {value:.3e} at pivot {s} (column {c})"),
                _ => write!(f, "numerically singular: reciprocal condition estimate {value:.3e}"),
            },
            Error::Singular { step } => write!(f, "singular: no pivot at step {step}"),
            Error::NonFiniteSolution => write!(f, "the solution has non-finite entries"),
            Error::OutOfMemory => write!(f, "out of memory"),
            Error::Unsupported(s) => write!(f, "unsupported: {s}"),
            Error::Internal(s) => write!(f, "internal error: {s}"),
        }
    }
}

impl std::error::Error for Error {}

/// What the last analysis and factorization did.
#[derive(Clone, Debug, Default)]
pub struct Info {
    pub n: usize,
    pub nnz_a: usize,
    /// stored entries of the factor (LL^T / LDL^T); 0 for LU
    pub nnz_factor: usize,
    pub supernodal: bool,
    /// the last update analyzed the pattern (it was new to this solver)
    pub reanalyzed: bool,
    /// the analysis came from another solver with the same pattern
    pub analysis_shared: bool,
    pub analyses: usize,
    pub factorizations: usize,
    /// LDL^T: number of negative pivots
    pub neg_pivots: usize,
    pub min_pivot: f64,
    pub max_pivot: f64,
    pub pivot_ratio: f64,
    /// LU with the rcond check: the estimate
    pub rcond: Option<f64>,
    pub fail_step: Option<usize>,
    pub fail_column: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AnalysisKey {
    lu: bool,
    ordering: Ordering,
    nodal: Nodal,
}

pub(crate) enum Sym {
    Chol(SymbolicCholesky<usize>),
    Lu(SymbolicLu<usize>),
}

pub(crate) struct Analysis {
    pub(crate) key: AnalysisKey,
    pub(crate) n: usize,
    pub(crate) col_ptr: Vec<usize>,
    pub(crate) row_idx: Vec<usize>,
    pub(crate) sym: Sym,
}

enum Numeric {
    Chol(Vec<f64>),
    Lu(NumericLu<usize, f64>),
}

struct Threads {
    par: Par,
    pool: Option<rayon::ThreadPool>,
}

impl Threads {
    fn new(n: usize) -> Result<Self, Error> {
        match n {
            1 => Ok(Threads { par: Par::Seq, pool: None }),
            0 => Ok(Threads { par: Par::rayon(0), pool: None }),
            t => {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .build()
                    .map_err(|e| Error::Internal(format!("thread pool: {e}")))?;
                Ok(Threads { par: Par::rayon(t), pool: Some(pool) })
            }
        }
    }

    fn run<R: Send>(&self, f: impl FnOnce(Par) -> R + Send) -> R {
        let par = self.par;
        match &self.pool {
            Some(p) => p.install(move || f(par)),
            None => f(par),
        }
    }
}

/// One matrix at a time: analyze its pattern, factorize its values, solve.
pub struct Solver {
    opts: Options,
    analysis: Option<Arc<Analysis>>,
    numeric: Option<Numeric>,
    anorm1: f64,
    info: Info,
    factor_threads: Threads,
    solve_threads: Threads,
    workspace: Option<(StackReq, MemBuffer)>,
}

fn threshold(n: Nodal) -> SupernodalThreshold {
    match n {
        Nodal::Auto => SupernodalThreshold::AUTO,
        Nodal::Simplicial => SupernodalThreshold::FORCE_SIMPLICIAL,
        Nodal::Supernodal => SupernodalThreshold::FORCE_SUPERNODAL,
    }
}

fn faer_err(e: faer::sparse::FaerError) -> Error {
    match e {
        faer::sparse::FaerError::OutOfMemory => Error::OutOfMemory,
        e => Error::Internal(format!("{e:?}")),
    }
}

/// Check a compressed-column (or -row) pattern: sizes, ranges, sorted rows
/// without duplicates, and (symmetric kinds) a diagonal entry in every column.
pub fn validate_pattern(n: usize, col_ptr: &[usize], row_idx: &[usize], need_diag: bool) -> Result<(), Error> {
    let Some(n1) = n.checked_add(1) else {
        return Err(Error::InvalidInput("n too large".into()));
    };
    if col_ptr.len() != n1 || col_ptr[0] != 0 || col_ptr[n] != row_idx.len() {
        return Err(Error::InvalidInput("pointers do not match n and the number of entries".into()));
    }
    for j in 0..n {
        let (a, b) = (col_ptr[j], col_ptr[j + 1]);
        if b < a || b > row_idx.len() {
            return Err(Error::InvalidInput(format!("pointers decrease or pass the number of entries at {j}")));
        }
        let mut has_diag = false;
        for p in a..b {
            let i = row_idx[p];
            if i >= n {
                return Err(Error::InvalidInput(format!("index {i} out of range in {j}")));
            }
            if p > a && row_idx[p - 1] >= i {
                return Err(Error::InvalidInput(format!("indices of {j} unsorted or duplicated")));
            }
            has_diag |= i == j;
        }
        if need_diag && !has_diag {
            return Err(Error::StructurallySingular { column: j });
        }
    }
    Ok(())
}

#[cfg(feature = "nd")]
fn nd_permutation(n: usize, col_ptr: &[usize], row_idx: &[usize]) -> Result<(Vec<usize>, Vec<usize>), Error> {
    // the ordering wants the full symmetric pattern: union with the transpose
    let mut cols: Vec<Vec<i32>> = vec![Vec::new(); n];
    for j in 0..n {
        for &i in &row_idx[col_ptr[j]..col_ptr[j + 1]] {
            cols[j].push(i as i32);
            if i != j {
                cols[i].push(j as i32);
            }
        }
    }
    let mut cp = Vec::with_capacity(n + 1);
    let mut ri = Vec::new();
    cp.push(0i32);
    for c in cols.iter_mut() {
        c.sort_unstable();
        c.dedup();
        ri.extend_from_slice(c);
        cp.push(ri.len() as i32);
    }
    let pat = feral_metis::CscPattern::new(n, &cp, &ri)
        .ok_or_else(|| Error::InvalidInput("pattern rejected by the ordering".into()))?;
    let p = feral_metis::metis_order(&pat).map_err(|e| Error::Internal(format!("ordering: {e:?}")))?;
    let fwd: Vec<usize> = p.iter().map(|&v| v as usize).collect();
    let mut inv = vec![usize::MAX; n];
    for (i, &f) in fwd.iter().enumerate() {
        if f >= n || inv[f] != usize::MAX {
            return Err(Error::Internal("ordering returned no permutation".into()));
        }
        inv[f] = i;
    }
    Ok((fwd, inv))
}

impl Analysis {
    fn new(opts: &Options, n: usize, col_ptr: &[usize], row_idx: &[usize]) -> Result<Analysis, Error> {
        let key = AnalysisKey { lu: opts.kind == Kind::Lu, ordering: opts.ordering, nodal: opts.nodal };
        let a = SymbolicSparseColMatRef::new_checked(n, n, col_ptr, None, row_idx);
        let sym = if key.lu {
            let params = LuSymbolicParams { supernodal_flop_ratio_threshold: threshold(opts.nodal), ..Default::default() };
            Sym::Lu(factorize_symbolic_lu(a, params).map_err(faer_err)?)
        } else {
            let params = CholeskySymbolicParams { supernodal_flop_ratio_threshold: threshold(opts.nodal), ..Default::default() };
            let s = match opts.ordering {
                Ordering::Amd => factorize_symbolic_cholesky(a, Side::Lower, SymmetricOrdering::Amd, params),
                Ordering::NestedDissection => {
                    #[cfg(feature = "nd")]
                    {
                        let (fwd, inv) = nd_permutation(n, col_ptr, row_idx)?;
                        let perm = faer::perm::PermRef::new_checked(&fwd, &inv, n);
                        factorize_symbolic_cholesky(a, Side::Lower, SymmetricOrdering::Custom(perm), params)
                    }
                    #[cfg(not(feature = "nd"))]
                    {
                        return Err(Error::Unsupported("nested dissection needs the `nd` feature".into()));
                    }
                }
            };
            Sym::Chol(s.map_err(faer_err)?)
        };
        Ok(Analysis { key, n, col_ptr: col_ptr.to_vec(), row_idx: row_idx.to_vec(), sym })
    }

    fn same_pattern(&self, n: usize, col_ptr: &[usize], row_idx: &[usize]) -> bool {
        self.n == n && self.col_ptr.as_slice() == col_ptr && self.row_idx.as_slice() == row_idx
    }
}

struct Pivots {
    min: f64,
    max: f64,
    amin: f64,
    amax: f64,
    amin_step: usize,
    neg: usize,
}

/// pivots in elimination order: L_jj^2 for LL^T, D_jj for LDL^T
fn pivots(s: &SymbolicCholesky<usize>, l: &[f64], llt: bool) -> Pivots {
    let mut p = Pivots { min: f64::INFINITY, max: f64::NEG_INFINITY, amin: f64::INFINITY, amax: 0.0, amin_step: 0, neg: 0 };
    let mut step = 0usize;
    let mut push = |v: f64| {
        let d = if llt { v * v } else { v };
        p.min = p.min.min(d);
        p.max = p.max.max(d);
        if d.abs() < p.amin {
            p.amin = d.abs();
            p.amin_step = step;
        }
        p.amax = p.amax.max(d.abs());
        if d < 0.0 {
            p.neg += 1;
        }
        step += 1;
    };
    match s.raw() {
        SymbolicCholeskyRaw::Simplicial(t) => {
            // the diagonal is the first entry of each column of L
            let cp = t.col_ptr();
            for j in 0..t.nrows() {
                push(l[cp[j]]);
            }
        }
        SymbolicCholeskyRaw::Supernodal(t) => {
            let (begin, end) = (t.supernode_begin(), t.supernode_end());
            let (cpv, nnzs) = (t.col_ptr_for_val(), t.nnz_per_super());
            for s in 0..t.n_supernodes() {
                let ncols = end[s] - begin[s];
                let nrows = ncols + nnzs[s];
                for j in 0..ncols {
                    push(l[cpv[s] + j * nrows + j]);
                }
            }
        }
    }
    p
}

impl Solver {
    pub fn new(opts: Options) -> Result<Solver, Error> {
        if !(opts.symmetry_tol >= 0.0 && opts.symmetry_tol.is_finite()) {
            return Err(Error::InvalidInput("symmetry_tol must be finite and >= 0".into()));
        }
        for (name, v) in [("singular_pivot_ratio", opts.singular_pivot_ratio), ("min_rcond", opts.min_rcond)] {
            if let Some(v) = v {
                if !(v >= 0.0 && v.is_finite()) {
                    return Err(Error::InvalidInput(format!("{name} must be finite and >= 0")));
                }
            }
        }
        #[cfg(not(feature = "nd"))]
        if opts.ordering == Ordering::NestedDissection && opts.kind != Kind::Lu {
            return Err(Error::Unsupported("nested dissection needs the `nd` feature".into()));
        }
        Ok(Solver {
            factor_threads: Threads::new(opts.factor_threads)?,
            solve_threads: Threads::new(opts.solve_threads)?,
            opts,
            analysis: None,
            numeric: None,
            anorm1: 0.0,
            info: Info::default(),
            workspace: None,
        })
    }

    pub fn options(&self) -> &Options {
        &self.opts
    }

    pub fn info(&self) -> &Info {
        &self.info
    }

    /// Drop the factor and keep the analysis: a solve fails with
    /// [`Error::NotFactorized`] until the next successful factorization.
    /// For callers that reject an input before reaching `update`.
    pub fn drop_factor(&mut self) {
        self.numeric = None;
    }

    /// A factor from the last successful factorization is in place.
    pub fn is_factorized(&self) -> bool {
        self.numeric.is_some()
    }

    fn fail(&mut self, e: Error) -> Error {
        self.numeric = None;
        let (step, column) = match &e {
            Error::NotPositiveDefinite { step, column } | Error::ZeroPivot { step, column } => (Some(*step), Some(*column)),
            Error::NumericallySingular { step, column, .. } => (*step, *column),
            Error::StructurallySingular { column } => (None, Some(*column)),
            Error::Singular { step } => (Some(*step), None),
            _ => (None, None),
        };
        self.info.fail_step = step;
        self.info.fail_column = column;
        e
    }

    /// Symbolic analysis of a pattern: ordering and the factor's structure.
    /// Drops any factor.
    pub fn analyze(&mut self, n: usize, col_ptr: &[usize], row_idx: &[usize]) -> Result<(), Error> {
        self.numeric = None;
        self.analysis = None;
        self.info = Info { analyses: self.info.analyses, factorizations: self.info.factorizations, ..Info::default() };
        let r = (|| {
            validate_pattern(n, col_ptr, row_idx, self.opts.kind != Kind::Lu)?;
            let key = AnalysisKey { lu: self.opts.kind == Kind::Lu, ordering: self.opts.ordering, nodal: self.opts.nodal };
            if self.opts.share_analysis {
                if let Some(a) = cache::find(key, n, col_ptr, row_idx) {
                    return Ok((a, true));
                }
            }
            let a = Arc::new(Analysis::new(&self.opts, n, col_ptr, row_idx)?);
            if self.opts.share_analysis {
                cache::put(a.clone());
            }
            Ok((a, false))
        })();
        let (a, shared) = r.map_err(|e| self.fail(e))?;
        self.info.n = n;
        self.info.nnz_a = row_idx.len();
        self.info.analysis_shared = shared;
        self.info.analyses += !shared as usize;
        if let Sym::Chol(s) = &a.sym {
            self.info.nnz_factor = s.len_val();
            self.info.supernodal = matches!(s.raw(), SymbolicCholeskyRaw::Supernodal(_));
        }
        self.analysis = Some(a);
        Ok(())
    }

    fn check_values(&self, a: &Analysis, values: &[f64]) -> Result<(), Error> {
        if values.len() != a.row_idx.len() {
            return Err(Error::InvalidInput("values do not match the number of entries".into()));
        }
        if let Some(p) = values.iter().position(|v| !v.is_finite()) {
            return Err(Error::NonFiniteInput { position: p });
        }
        if self.opts.kind != Kind::Lu && self.opts.check_symmetry {
            // every off-diagonal entry against its mirror; a mirror that is not
            // stored counts as zero, so an entry on either side of the diagonal
            // without a counterpart is refused
            let (cp, ri) = (&a.col_ptr, &a.row_idx);
            for j in 0..a.n {
                for p in cp[j]..cp[j + 1] {
                    let i = ri[p];
                    if i == j {
                        continue;
                    }
                    let col = &ri[cp[i]..cp[i + 1]];
                    let mirror = col.binary_search(&j).ok().map(|q| values[cp[i] + q]);
                    // a pair with both entries stored is compared once, from below
                    if mirror.is_some() && i < j {
                        continue;
                    }
                    let other = mirror.unwrap_or(0.0);
                    let v = values[p];
                    if (v - other).abs() > self.opts.symmetry_tol * v.abs().max(other.abs()) {
                        // position in the caller's matrix: compressed rows store A^T
                        let (row, col) = if self.opts.compressed_rows { (j, i) } else { (i, j) };
                        return Err(Error::NotSymmetric { row, col });
                    }
                }
            }
        }
        Ok(())
    }

    /// Numeric factorization of values on the analyzed pattern. The pattern
    /// must be the analyzed one entry for entry.
    pub fn factorize(&mut self, col_ptr: &[usize], row_idx: &[usize], values: &[f64]) -> Result<&Info, Error> {
        self.numeric = None;
        self.info.fail_step = None;
        self.info.fail_column = None;
        let a = match &self.analysis {
            Some(a) => a.clone(),
            None => return Err(self.fail(Error::NotAnalyzed)),
        };
        if !a.same_pattern(a.n, col_ptr, row_idx) || col_ptr.len() != a.n + 1 {
            return Err(self.fail(Error::PatternChanged));
        }
        if let Err(e) = self.check_values(&a, values) {
            return Err(self.fail(e));
        }
        match self.factorize_numeric(&a, values) {
            Ok(()) => {
                self.info.factorizations += 1;
                Ok(&self.info)
            }
            Err(e) => Err(self.fail(e)),
        }
    }

    /// Analyze when the pattern is new to this solver, then factorize.
    /// `info().reanalyzed` tells which path was taken.
    pub fn update(&mut self, n: usize, col_ptr: &[usize], row_idx: &[usize], values: &[f64]) -> Result<&Info, Error> {
        let fresh = !matches!(&self.analysis, Some(a) if a.same_pattern(n, col_ptr, row_idx));
        if fresh {
            self.analyze(n, col_ptr, row_idx)?;
        }
        self.factorize(col_ptr, row_idx, values)?;
        self.info.reanalyzed = fresh;
        Ok(&self.info)
    }

    fn factorize_numeric(&mut self, a: &Analysis, values: &[f64]) -> Result<(), Error> {
        let n = a.n;
        let mat = SparseColMatRef::new(SymbolicSparseColMatRef::new_checked(n, n, &a.col_ptr, None, &a.row_idx), values);
        // the norm of the matrix the factor stands for: the symmetric kinds read
        // the lower triangle of the column view only
        self.anorm1 = if self.opts.kind != Kind::Lu {
            norm1_symmetric_lower(n, &a.col_ptr, &a.row_idx, values)
        } else if self.opts.compressed_rows {
            norm_inf(n, &a.col_ptr, &a.row_idx, values)
        } else {
            norm1(n, &a.col_ptr, values)
        };
        self.info.rcond = None;
        match &a.sym {
            Sym::Chol(s) => {
                let llt = self.opts.kind == Kind::Llt;
                let mut l = Vec::new();
                l.try_reserve_exact(s.len_val()).map_err(|_| Error::OutOfMemory)?;
                l.resize(s.len_val(), 0.0f64);
                // faer reports the failing pivot 1-based from the simplicial
                // kernel and 0-based from the supernodal one
                let supernodal = matches!(s.raw(), SymbolicCholeskyRaw::Supernodal(_));
                let step0 = |index: usize| if supernodal { index } else { index.saturating_sub(1) };
                let column = |step: usize| s.perm().map(|p| p.arrays().0[step]).unwrap_or(step);
                let r: Result<(), Error> = self.factor_threads.run(|par| {
                    if llt {
                        let req = s.factorize_numeric_llt_scratch::<f64>(par, Default::default());
                        let mut buf = MemBuffer::try_new(req).map_err(|_| Error::OutOfMemory)?;
                        s.factorize_numeric_llt(&mut l, mat, Side::Lower, LltRegularization::default(), par, MemStack::new(&mut buf), Default::default())
                            .map(|_| ())
                            .map_err(|e| {
                                let faer::linalg::solvers::LltError::NonPositivePivot { index } = e;
                                let st = step0(index);
                                Error::NotPositiveDefinite { step: st, column: column(st) }
                            })
                    } else {
                        let req = s.factorize_numeric_ldlt_scratch::<f64>(par, Default::default());
                        let mut buf = MemBuffer::try_new(req).map_err(|_| Error::OutOfMemory)?;
                        s.factorize_numeric_ldlt(&mut l, mat, Side::Lower, LdltRegularization::default(), par, MemStack::new(&mut buf), Default::default())
                            .map(|_| ())
                            .map_err(|e| {
                                let faer::linalg::cholesky::ldlt::factor::LdltError::ZeroPivot { index } = e;
                                let st = step0(index);
                                Error::ZeroPivot { step: st, column: column(st) }
                            })
                    }
                });
                r?;
                let pv = pivots(s, &l, llt);
                self.info.min_pivot = pv.min;
                self.info.max_pivot = pv.max;
                self.info.pivot_ratio = pv.amin / pv.amax;
                self.info.neg_pivots = pv.neg;
                let thr = self.opts.singular_pivot_ratio.unwrap_or(n as f64 * f64::EPSILON);
                if !(self.info.pivot_ratio >= thr) {
                    return Err(Error::NumericallySingular {
                        step: Some(pv.amin_step),
                        column: Some(column(pv.amin_step)),
                        value: self.info.pivot_ratio,
                    });
                }
                self.numeric = Some(Numeric::Chol(l));
            }
            Sym::Lu(s) => {
                let mut numeric = NumericLu::new();
                let r: Result<(), Error> = self.factor_threads.run(|par| {
                    let req = s.factorize_numeric_lu_scratch::<f64>(par, Default::default());
                    let mut buf = MemBuffer::try_new(req).map_err(|_| Error::OutOfMemory)?;
                    s.factorize_numeric_lu(&mut numeric, mat, par, MemStack::new(&mut buf), Default::default())
                        .map(|_| ())
                        .map_err(|e| match e {
                            LuError::SymbolicSingular { index } => Error::Singular { step: index },
                            LuError::Generic(g) => faer_err(g),
                        })
                });
                r?;
                self.numeric = Some(Numeric::Lu(numeric));
                let thr = self.opts.min_rcond.unwrap_or(f64::EPSILON);
                if thr > 0.0 {
                    let rc = match self.rcond_estimate() {
                        Ok(rc) => rc,
                        // a non-finite probe solve is a singular factor too
                        Err(Error::NonFiniteSolution) => 0.0,
                        Err(e) => return Err(e),
                    };
                    self.info.rcond = Some(rc);
                    if !(rc >= thr) {
                        return Err(Error::NumericallySingular { step: None, column: None, value: rc });
                    }
                }
            }
        }
        Ok(())
    }

    /// Solve A X = B in place. `b` holds `nrhs` columns of length n, column j
    /// starting at `j * ldb`.
    pub fn solve_strided(&mut self, b: &mut [f64], nrhs: usize, ldb: usize) -> Result<(), Error> {
        self.solve_impl(b, nrhs, ldb, false)
    }

    /// Solve A X = B in place for a contiguous column-major block.
    pub fn solve(&mut self, b: &mut [f64], nrhs: usize) -> Result<(), Error> {
        let n = self.info.n;
        self.solve_impl(b, nrhs, n, false)
    }

    fn solve_impl(&mut self, b: &mut [f64], nrhs: usize, ldb: usize, transpose: bool) -> Result<(), Error> {
        let (a, num) = match (&self.analysis, &self.numeric) {
            (Some(a), Some(num)) => (a.clone(), num),
            _ => return Err(Error::NotFactorized),
        };
        let n = a.n;
        let need = if nrhs == 0 { Some(0) } else { ldb.checked_mul(nrhs - 1).and_then(|v| v.checked_add(n)) };
        match need {
            Some(len) if ldb >= n && b.len() >= len => {}
            _ => return Err(Error::InvalidInput("right-hand side shorter than ldb * (nrhs - 1) + n".into())),
        }
        if nrhs == 0 || n == 0 {
            return Ok(());
        }
        // the arrays describe A^T when they are compressed rows
        let trans = transpose ^ self.opts.compressed_rows;
        let par = self.solve_threads.par;
        let req = match &a.sym {
            Sym::Chol(s) => s.solve_in_place_scratch::<f64>(nrhs, par),
            Sym::Lu(s) => {
                if trans {
                    s.solve_transpose_in_place_scratch::<f64>(nrhs, par)
                } else {
                    s.solve_in_place_scratch::<f64>(nrhs, par)
                }
            }
        };
        // dyn-stack marks an overflowing request with alignment 0
        if req.align_bytes() == 0 {
            return Err(Error::OutOfMemory);
        }
        // the requirement depends on the factor and on n as well as on nrhs:
        // reuse the buffer only when it covers this one
        let fits = matches!(&self.workspace, Some((have, _))
            if have.size_bytes() >= req.size_bytes() && have.align_bytes() >= req.align_bytes());
        if !fits {
            self.workspace = None;
            let buf = MemBuffer::try_new(req).map_err(|_| Error::OutOfMemory)?;
            self.workspace = Some((req, buf));
        }
        let (_, buf) = self.workspace.as_mut().unwrap();
        let rhs = MatMut::from_column_major_slice_with_stride_mut(&mut b[..ldb * (nrhs - 1) + n], n, nrhs, ldb);
        let kind = self.opts.kind;
        let pool = &self.solve_threads;
        pool.run(|par| match (&a.sym, num) {
            (Sym::Chol(s), Numeric::Chol(l)) => {
                if kind == Kind::Llt {
                    LltRef::<usize, f64>::new(s, l).solve_in_place_with_conj(Conj::No, rhs, par, MemStack::new(buf));
                } else {
                    LdltRef::<usize, f64>::new(s, l).solve_in_place_with_conj(Conj::No, rhs, par, MemStack::new(buf));
                }
            }
            (Sym::Lu(s), Numeric::Lu(nu)) => {
                let lu = LuRef::<usize, f64>::new_unchecked(s, nu);
                if trans {
                    lu.solve_transpose_in_place_with_conj(Conj::No, rhs, par, MemStack::new(buf));
                } else {
                    lu.solve_in_place_with_conj(Conj::No, rhs, par, MemStack::new(buf));
                }
            }
            _ => unreachable!("analysis and factor kinds always match"),
        });
        for j in 0..nrhs {
            if b[j * ldb..j * ldb + n].iter().any(|v| !v.is_finite()) {
                return Err(Error::NonFiniteSolution);
            }
        }
        Ok(())
    }

    /// Hager–Higham estimate of 1 / (||A||_1 ||A^-1||_1) with the current
    /// factor (a handful of solves). W. W. Hager, Condition estimates, SIAM
    /// J. Sci. Stat. Comput. 5 (1984) 311–316; N. J. Higham, Algorithm 674,
    /// ACM Trans. Math. Softw. 14 (1988) 381–396.
    pub fn rcond_estimate(&mut self) -> Result<f64, Error> {
        let n = self.info.n;
        if self.numeric.is_none() {
            return Err(Error::NotFactorized);
        }
        if n == 0 {
            return Ok(1.0);
        }
        let symmetric = self.opts.kind != Kind::Lu;
        let mut x = vec![1.0 / n as f64; n];
        let mut y = x.clone();
        self.solve_impl(&mut y, 1, n, false)?;
        let mut est = y.iter().map(|v| v.abs()).sum::<f64>();
        for _ in 0..5 {
            let mut z: Vec<f64> = y.iter().map(|v| if *v >= 0.0 { 1.0 } else { -1.0 }).collect();
            self.solve_impl(&mut z, 1, n, !symmetric)?;
            let (mut j, mut zmax) = (0, 0.0f64);
            for (i, v) in z.iter().enumerate() {
                if v.abs() > zmax {
                    zmax = v.abs();
                    j = i;
                }
            }
            let ztx: f64 = z.iter().zip(&x).map(|(a, b)| a * b).sum();
            if zmax <= ztx {
                break;
            }
            x = vec![0.0; n];
            x[j] = 1.0;
            y = x.clone();
            self.solve_impl(&mut y, 1, n, false)?;
            let e = y.iter().map(|v| v.abs()).sum::<f64>();
            if e <= est {
                break;
            }
            est = e;
        }
        // Higham's safeguard against the estimator's worst cases
        let mut w: Vec<f64> = (0..n)
            .map(|i| {
                let s = if i % 2 == 0 { 1.0 } else { -1.0 };
                s * (1.0 + i as f64 / (n.max(2) - 1) as f64)
            })
            .collect();
        self.solve_impl(&mut w, 1, n, false)?;
        est = est.max(2.0 * w.iter().map(|v| v.abs()).sum::<f64>() / (3.0 * n as f64));
        Ok(1.0 / (self.anorm1 * est))
    }
}

/// max column sum of |a| over the stored arrays
fn norm1(n: usize, col_ptr: &[usize], values: &[f64]) -> f64 {
    (0..n).map(|j| values[col_ptr[j]..col_ptr[j + 1]].iter().map(|v| v.abs()).sum::<f64>()).fold(0.0, f64::max)
}

/// 1-norm of the symmetric matrix whose lower triangle (rows i >= j of each
/// column j) is stored; entries above the diagonal are not part of it
fn norm1_symmetric_lower(n: usize, col_ptr: &[usize], row_idx: &[usize], values: &[f64]) -> f64 {
    let mut c = vec![0.0f64; n];
    for j in 0..n {
        for p in col_ptr[j]..col_ptr[j + 1] {
            let i = row_idx[p];
            if i < j {
                continue;
            }
            c[j] += values[p].abs();
            if i != j {
                c[i] += values[p].abs();
            }
        }
    }
    c.into_iter().fold(0.0, f64::max)
}

/// max row sum of |a| over the stored arrays
fn norm_inf(n: usize, col_ptr: &[usize], row_idx: &[usize], values: &[f64]) -> f64 {
    let mut r = vec![0.0f64; n];
    for j in 0..n {
        for p in col_ptr[j]..col_ptr[j + 1] {
            r[row_idx[p]] += values[p].abs();
        }
    }
    r.into_iter().fold(0.0, f64::max)
}
