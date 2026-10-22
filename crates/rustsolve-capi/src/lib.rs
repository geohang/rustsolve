//! C ABI of rustsolve; the declarations are in `include/rustsolve.h`.
//!
//! Every entry point returns a status (`RS_OK` = 0). A panic is caught at the
//! boundary and returned as `RS_PANIC`. A failed update leaves the handle
//! without a factor: a later `rs_solve` returns `RS_NOT_FACTORIZED`, never a
//! solution of an earlier matrix.

use std::ffi::{c_char, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

use rustsolve::{Error, Kind, Nodal, Options, Ordering, Solver};

pub const RS_ABI_VERSION: u32 = 1;

pub const RS_OK: i32 = 0;
pub const RS_INVALID_INPUT: i32 = 1;
pub const RS_STRUCTURALLY_SINGULAR: i32 = 2;
pub const RS_NOT_SYMMETRIC: i32 = 3;
pub const RS_NON_FINITE_INPUT: i32 = 4;
pub const RS_PATTERN_CHANGED: i32 = 5;
pub const RS_NOT_ANALYZED: i32 = 6;
pub const RS_NOT_FACTORIZED: i32 = 7;
pub const RS_NOT_POSITIVE_DEFINITE: i32 = 8;
pub const RS_ZERO_PIVOT: i32 = 9;
pub const RS_NUMERICALLY_SINGULAR: i32 = 10;
pub const RS_SINGULAR: i32 = 11;
pub const RS_NON_FINITE_SOLUTION: i32 = 12;
pub const RS_OUT_OF_MEMORY: i32 = 13;
pub const RS_INTERNAL: i32 = 14;
pub const RS_PANIC: i32 = 15;
pub const RS_UNSUPPORTED: i32 = 16;

/// `rs_options` in rustsolve.h.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RsOptions {
    pub kind: i32,
    pub ordering: i32,
    pub nodal: i32,
    pub threads: i32,
    pub solve_threads: i32,
    pub check_symmetry: i32,
    pub compressed_rows: i32,
    pub share_analysis: i32,
    pub symmetry_tol: f64,
    pub singular_pivot_ratio: f64,
    pub min_rcond: f64,
}

/// `rs_info` in rustsolve.h.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RsInfo {
    pub n: i64,
    pub nnz_a: i64,
    pub nnz_factor: i64,
    pub supernodal: i32,
    pub reanalyzed: i32,
    pub analysis_shared: i32,
    pub reserved: i32,
    pub analyses: i64,
    pub factorizations: i64,
    pub neg_pivots: i64,
    pub min_pivot: f64,
    pub max_pivot: f64,
    pub pivot_ratio: f64,
    pub rcond: f64,
    pub fail_step: i64,
    pub fail_column: i64,
}

pub struct RsSolver {
    solver: Solver,
    cp: Vec<usize>,
    ri: Vec<usize>,
    err: CString,
}

fn guard(f: impl FnOnce() -> i32) -> i32 {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(RS_PANIC)
}

impl RsSolver {
    fn status(&mut self, r: Result<(), Error>) -> i32 {
        match r {
            Ok(()) => {
                self.err = CString::default();
                RS_OK
            }
            Err(e) => {
                self.err = CString::new(format!("{}: {e}", e.name())).unwrap_or_default();
                e.code()
            }
        }
    }
}

fn default_options() -> RsOptions {
    RsOptions {
        kind: 0,
        ordering: 0,
        nodal: 0,
        threads: 1,
        solve_threads: 1,
        check_symmetry: 1,
        compressed_rows: 0,
        share_analysis: 0,
        symmetry_tol: 1e-12,
        singular_pivot_ratio: -1.0,
        min_rcond: -1.0,
    }
}

fn to_options(o: &RsOptions) -> Result<Options, i32> {
    let kind = match o.kind {
        0 => Kind::Llt,
        1 => Kind::Ldlt,
        2 => Kind::Lu,
        _ => return Err(RS_INVALID_INPUT),
    };
    let ordering = match o.ordering {
        0 => Ordering::Amd,
        1 => Ordering::NestedDissection,
        _ => return Err(RS_INVALID_INPUT),
    };
    let nodal = match o.nodal {
        0 => Nodal::Auto,
        1 => Nodal::Simplicial,
        2 => Nodal::Supernodal,
        _ => return Err(RS_INVALID_INPUT),
    };
    if o.threads < 0 || o.solve_threads < 0 {
        return Err(RS_INVALID_INPUT);
    }
    // non-finite first: a negative value is the default sentinel, and -inf
    // must not pass for one
    if !(o.singular_pivot_ratio.is_finite() && o.min_rcond.is_finite() && o.symmetry_tol.is_finite()) {
        return Err(RS_INVALID_INPUT);
    }
    let opt = |v: f64| if v < 0.0 { None } else { Some(v) };
    Ok(Options {
        kind,
        ordering,
        nodal,
        check_symmetry: o.check_symmetry != 0,
        symmetry_tol: o.symmetry_tol,
        compressed_rows: o.compressed_rows != 0,
        singular_pivot_ratio: opt(o.singular_pivot_ratio),
        min_rcond: opt(o.min_rcond),
        factor_threads: o.threads as usize,
        solve_threads: o.solve_threads as usize,
        share_analysis: o.share_analysis != 0,
    })
}

#[no_mangle]
pub extern "C" fn rs_abi_version() -> u32 {
    RS_ABI_VERSION
}

/// # Safety
/// `opt` must point to writable memory for one `rs_options`.
#[no_mangle]
pub unsafe extern "C" fn rs_default_options(opt: *mut RsOptions) -> i32 {
    guard(|| {
        if opt.is_null() {
            return RS_INVALID_INPUT;
        }
        *opt = default_options();
        RS_OK
    })
}

/// # Safety
/// `opt` (or null for the defaults) and `out` must be valid pointers.
#[no_mangle]
pub unsafe extern "C" fn rs_new(opt: *const RsOptions, out: *mut *mut RsSolver) -> i32 {
    guard(|| {
        if out.is_null() {
            return RS_INVALID_INPUT;
        }
        *out = std::ptr::null_mut();
        let o = if opt.is_null() { default_options() } else { *opt };
        let opts = match to_options(&o) {
            Ok(v) => v,
            Err(c) => return c,
        };
        match Solver::new(opts) {
            Ok(solver) => {
                *out = Box::into_raw(Box::new(RsSolver { solver, cp: vec![], ri: vec![], err: CString::default() }));
                RS_OK
            }
            Err(e) => e.code(),
        }
    })
}

/// # Safety
/// `s` must come from `rs_new` (or be null) and not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn rs_free(s: *mut RsSolver) {
    if !s.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(s))));
    }
}

trait Idx: Copy {
    fn get(self) -> Option<usize>;
}
impl Idx for i32 {
    fn get(self) -> Option<usize> {
        usize::try_from(self).ok()
    }
}
impl Idx for i64 {
    fn get(self) -> Option<usize> {
        usize::try_from(self).ok()
    }
}

/// `len` elements of `T` can form a slice (at most isize::MAX bytes)
fn slice_ok<T>(len: usize) -> bool {
    len.checked_mul(std::mem::size_of::<T>()).is_some_and(|b| b <= isize::MAX as usize)
}

unsafe fn update<T: Idx>(s: *mut RsSolver, n: i64, col_ptr: *const T, row_idx: *const T, values: *const f64) -> i32 {
    guard(|| {
        if s.is_null() {
            return RS_INVALID_INPUT;
        }
        let h = &mut *s;
        // whatever happens below, the previous matrix's factor is gone: an
        // input rejected here must not leave it for the next rs_solve
        h.solver.drop_factor();
        if col_ptr.is_null() || row_idx.is_null() || values.is_null() || n < 0 || n == i64::MAX {
            return h.status(Err(Error::InvalidInput("null pointer or invalid n".into())));
        }
        let n = n as usize;
        if !slice_ok::<T>(n + 1) {
            return h.status(Err(Error::InvalidInput("n too large".into())));
        }
        let cp = std::slice::from_raw_parts(col_ptr, n + 1);
        h.cp.clear();
        for &v in cp {
            match v.get() {
                Some(u) => h.cp.push(u),
                None => return h.status(Err(Error::InvalidInput("negative pointer".into()))),
            }
        }
        let nnz = h.cp[n];
        if !slice_ok::<T>(nnz) || !slice_ok::<f64>(nnz) {
            return h.status(Err(Error::InvalidInput("number of entries too large".into())));
        }
        let ri = std::slice::from_raw_parts(row_idx, nnz);
        h.ri.clear();
        for &v in ri {
            match v.get() {
                Some(u) => h.ri.push(u),
                None => return h.status(Err(Error::InvalidInput("negative index".into()))),
            }
        }
        let vals = std::slice::from_raw_parts(values, nnz);
        let (cp, ri) = (std::mem::take(&mut h.cp), std::mem::take(&mut h.ri));
        let r = h.solver.update(n, &cp, &ri, vals).map(|_| ());
        h.cp = cp;
        h.ri = ri;
        h.status(r)
    })
}

/// Analyze (only when the pattern differs from this handle's last one) and
/// factorize. 0-based compressed columns (rows with `compressed_rows`),
/// sorted, no duplicates.
///
/// # Safety
/// The arrays must hold n + 1, nnz and nnz entries, nnz = col_ptr[n].
#[no_mangle]
pub unsafe extern "C" fn rs_update_i32(s: *mut RsSolver, n: i64, col_ptr: *const i32, row_idx: *const i32, values: *const f64) -> i32 {
    update(s, n, col_ptr, row_idx, values)
}

/// As `rs_update_i32` with 64-bit indices.
///
/// # Safety
/// The arrays must hold n + 1, nnz and nnz entries, nnz = col_ptr[n].
#[no_mangle]
pub unsafe extern "C" fn rs_update_i64(s: *mut RsSolver, n: i64, col_ptr: *const i64, row_idx: *const i64, values: *const f64) -> i32 {
    update(s, n, col_ptr, row_idx, values)
}

/// Solve A X = B in place: `nrhs` columns of length n, column j at b + j * ldb.
///
/// # Safety
/// `b` must hold ldb * (nrhs - 1) + n values.
#[no_mangle]
pub unsafe extern "C" fn rs_solve(s: *mut RsSolver, nrhs: i64, b: *mut f64, ldb: i64) -> i32 {
    guard(|| {
        if s.is_null() {
            return RS_INVALID_INPUT;
        }
        let h = &mut *s;
        if nrhs < 0 || ldb < 0 || (nrhs > 0 && b.is_null()) {
            return h.status(Err(Error::InvalidInput("null pointer or negative size".into())));
        }
        let n = h.solver.info().n;
        let (nrhs, ldb) = (nrhs as usize, ldb as usize);
        if nrhs == 0 {
            let r = if h.solver.is_factorized() { Ok(()) } else { Err(Error::NotFactorized) };
            return h.status(r);
        }
        if ldb < n {
            return h.status(Err(Error::InvalidInput("ldb < n".into())));
        }
        let len = match ldb.checked_mul(nrhs - 1).and_then(|v| v.checked_add(n)) {
            Some(v) if slice_ok::<f64>(v) => v,
            _ => return h.status(Err(Error::InvalidInput("ldb * nrhs overflows".into()))),
        };
        let bs = std::slice::from_raw_parts_mut(b, len);
        let r = h.solver.solve_strided(bs, nrhs, ldb);
        h.status(r)
    })
}

/// Reciprocal 1-norm condition estimate with the current factor.
///
/// # Safety
/// `s` and `out` must be valid.
#[no_mangle]
pub unsafe extern "C" fn rs_rcond_estimate(s: *mut RsSolver, out: *mut f64) -> i32 {
    guard(|| {
        if s.is_null() || out.is_null() {
            return RS_INVALID_INPUT;
        }
        let h = &mut *s;
        match h.solver.rcond_estimate() {
            Ok(v) => {
                *out = v;
                h.status(Ok(()))
            }
            Err(e) => h.status(Err(e)),
        }
    })
}

/// # Safety
/// `s` and `info` must be valid.
#[no_mangle]
pub unsafe extern "C" fn rs_get_info(s: *const RsSolver, info: *mut RsInfo) -> i32 {
    guard(|| {
        if s.is_null() || info.is_null() {
            return RS_INVALID_INPUT;
        }
        let i = (*s).solver.info();
        *info = RsInfo {
            n: i.n as i64,
            nnz_a: i.nnz_a as i64,
            nnz_factor: i.nnz_factor as i64,
            supernodal: i.supernodal as i32,
            reanalyzed: i.reanalyzed as i32,
            analysis_shared: i.analysis_shared as i32,
            reserved: 0,
            analyses: i.analyses as i64,
            factorizations: i.factorizations as i64,
            neg_pivots: i.neg_pivots as i64,
            min_pivot: i.min_pivot,
            max_pivot: i.max_pivot,
            pivot_ratio: i.pivot_ratio,
            rcond: i.rcond.unwrap_or(f64::NAN),
            fail_step: i.fail_step.map(|v| v as i64).unwrap_or(-1),
            fail_column: i.fail_column.map(|v| v as i64).unwrap_or(-1),
        };
        RS_OK
    })
}

/// Text of the handle's last error ("" after a success); valid until the
/// next call on the handle.
///
/// # Safety
/// `s` must be valid or null.
#[no_mangle]
pub unsafe extern "C" fn rs_last_error(s: *const RsSolver) -> *const c_char {
    if s.is_null() {
        return c"null handle".as_ptr();
    }
    (*s).err.as_ptr()
}

/// Name of a status code (static string).
#[no_mangle]
pub extern "C" fn rs_status_name(status: i32) -> *const c_char {
    let s: &'static std::ffi::CStr = match status {
        RS_OK => c"OK",
        RS_INVALID_INPUT => c"INVALID_INPUT",
        RS_STRUCTURALLY_SINGULAR => c"STRUCTURALLY_SINGULAR",
        RS_NOT_SYMMETRIC => c"NOT_SYMMETRIC",
        RS_NON_FINITE_INPUT => c"NON_FINITE_INPUT",
        RS_PATTERN_CHANGED => c"PATTERN_CHANGED",
        RS_NOT_ANALYZED => c"NOT_ANALYZED",
        RS_NOT_FACTORIZED => c"NOT_FACTORIZED",
        RS_NOT_POSITIVE_DEFINITE => c"NOT_POSITIVE_DEFINITE",
        RS_ZERO_PIVOT => c"ZERO_PIVOT",
        RS_NUMERICALLY_SINGULAR => c"NUMERICALLY_SINGULAR",
        RS_SINGULAR => c"SINGULAR",
        RS_NON_FINITE_SOLUTION => c"NON_FINITE_SOLUTION",
        RS_OUT_OF_MEMORY => c"OUT_OF_MEMORY",
        RS_INTERNAL => c"INTERNAL",
        RS_PANIC => c"PANIC",
        RS_UNSUPPORTED => c"UNSUPPORTED",
        _ => c"UNKNOWN",
    };
    s.as_ptr()
}

/// Drop the analyses shared between handles (`share_analysis`).
#[no_mangle]
pub extern "C" fn rs_clear_shared_analyses() {
    let _ = catch_unwind(rustsolve::clear_shared_analyses);
}
