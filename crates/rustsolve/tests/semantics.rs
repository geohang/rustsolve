//! The interface rules on small finite-element-like matrices. Accuracy and
//! speed against reference solvers are measured in bench/ on systems from
//! real finite-element models, not here.

use rustsolve::{Error, Kind, Nodal, Options, Ordering, Solver};

/// Compressed columns, both triangles.
#[derive(Clone)]
struct Csc {
    n: usize,
    cp: Vec<usize>,
    ri: Vec<usize>,
    v: Vec<f64>,
}

impl Csc {
    fn from_triplets(n: usize, mut t: Vec<(usize, usize, f64)>) -> Csc {
        t.sort_by_key(|e| (e.1, e.0));
        let mut cp = vec![0usize; n + 1];
        let mut ri = Vec::new();
        let mut v: Vec<f64> = Vec::new();
        let mut last: Option<(usize, usize)> = None;
        for (i, j, x) in t {
            if last == Some((i, j)) {
                *v.last_mut().unwrap() += x;
                continue;
            }
            last = Some((i, j));
            ri.push(i);
            v.push(x);
            cp[j + 1] += 1;
        }
        for j in 0..n {
            cp[j + 1] += cp[j];
        }
        Csc { n, cp, ri, v }
    }

    fn matvec(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.n];
        for j in 0..self.n {
            for p in self.cp[j]..self.cp[j + 1] {
                y[self.ri[p]] += self.v[p] * x[j];
            }
        }
        y
    }

    fn dense(&self) -> Vec<Vec<f64>> {
        let mut a = vec![vec![0.0; self.n]; self.n];
        for j in 0..self.n {
            for p in self.cp[j]..self.cp[j + 1] {
                a[self.ri[p]][j] = self.v[p];
            }
        }
        a
    }

    fn transpose(&self) -> Csc {
        let mut t = Vec::new();
        for j in 0..self.n {
            for p in self.cp[j]..self.cp[j + 1] {
                t.push((j, self.ri[p], self.v[p]));
            }
        }
        Csc::from_triplets(self.n, t)
    }
}

/// P1 stiffness of a structured triangle grid (nx x ny nodes) with
/// conductivity `sigma(cell)`, plus `shift` on nodes of the first column
/// (a Dirichlet-like boundary term). `shift = 0` gives the pure Neumann
/// matrix, singular with a constant null space.
fn fem2d(nx: usize, ny: usize, sigma: impl Fn(usize) -> f64, shift: f64) -> Csc {
    let id = |i: usize, j: usize| j * nx + i;
    let mut t = Vec::new();
    let mut cell = 0;
    for j in 0..ny - 1 {
        for i in 0..nx - 1 {
            for tri in [[id(i, j), id(i + 1, j), id(i, j + 1)], [id(i + 1, j + 1), id(i, j + 1), id(i + 1, j)]] {
                // right isosceles triangle, unit legs: the right-angle vertex couples
                // to both others with -s/2, the other two not at all
                let s = sigma(cell);
                cell += 1;
                let (a, b, c) = (tri[0], tri[1], tri[2]);
                for (p, q, w) in [(a, a, s), (b, b, s / 2.0), (c, c, s / 2.0), (a, b, -s / 2.0), (b, a, -s / 2.0), (a, c, -s / 2.0), (c, a, -s / 2.0), (b, c, 0.0), (c, b, 0.0)] {
                    t.push((p, q, w));
                }
            }
        }
    }
    for j in 0..ny {
        t.push((id(0, j), id(0, j), shift));
    }
    Csc::from_triplets(nx * ny, t)
}

fn rhs(n: usize, k: usize) -> Vec<f64> {
    (0..n * k).map(|i| ((i * 7919) % 101) as f64 / 101.0 - 0.5).collect()
}

fn relres(a: &Csc, x: &[f64], b: &[f64]) -> f64 {
    let n = a.n;
    let mut worst: f64 = 0.0;
    for k in 0..b.len() / n {
        let ax = a.matvec(&x[k * n..(k + 1) * n]);
        let bk = &b[k * n..(k + 1) * n];
        let r: f64 = ax.iter().zip(bk).map(|(p, q)| (p - q).powi(2)).sum::<f64>().sqrt();
        let bn: f64 = bk.iter().map(|q| q * q).sum::<f64>().sqrt();
        worst = worst.max(r / bn);
    }
    worst
}

fn dense_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for k in 0..n {
        let p = (k..n).max_by(|&i, &j| a[i][k].abs().partial_cmp(&a[j][k].abs()).unwrap()).unwrap();
        a.swap(k, p);
        b.swap(k, p);
        for i in k + 1..n {
            let f = a[i][k] / a[k][k];
            for j in k..n {
                a[i][j] -= f * a[k][j];
            }
            b[i] -= f * b[k];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|j| a[i][j] * x[j]).sum();
        x[i] = (b[i] - s) / a[i][i];
    }
    x
}

fn opts(kind: Kind) -> Options {
    Options { kind, ..Options::default() }
}

fn solver(kind: Kind) -> Solver {
    Solver::new(opts(kind)).unwrap()
}

#[test]
fn kinds_agree_and_block_equals_columns() {
    let a = fem2d(23, 17, |c| if c % 7 == 0 { 1e-3 } else { 1.0 }, 1.0);
    let n = a.n;
    let b = rhs(n, 5);
    let mut sols = Vec::new();
    for kind in [Kind::Llt, Kind::Ldlt, Kind::Lu] {
        let mut s = solver(kind);
        s.update(n, &a.cp, &a.ri, &a.v).unwrap();
        let mut x = b.clone();
        s.solve(&mut x, 5).unwrap();
        assert!(relres(&a, &x, &b) < 1e-12, "{kind:?}");
        let mut xc = b.clone();
        for k in 0..5 {
            s.solve(&mut xc[k * n..(k + 1) * n], 1).unwrap();
        }
        let d = x.iter().zip(&xc).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max);
        assert!(d < 1e-12, "{kind:?}: block vs columns {d}");
        sols.push(x);
    }
    let scale = sols[0].iter().map(|v| v.abs()).fold(0.0, f64::max);
    for s in &sols[1..] {
        let d = s.iter().zip(&sols[0]).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max);
        assert!(d / scale < 1e-10);
    }
}

#[test]
fn both_kernels_and_strided_solve() {
    let a = fem2d(30, 30, |_| 1.0, 0.5);
    let n = a.n;
    let b = rhs(n, 3);
    for nodal in [Nodal::Simplicial, Nodal::Supernodal] {
        let mut s = Solver::new(Options { nodal, ..Options::default() }).unwrap();
        s.update(n, &a.cp, &a.ri, &a.v).unwrap();
        assert_eq!(s.info().supernodal, nodal == Nodal::Supernodal);
        // columns with a gap of 4 between them
        let ldb = n + 4;
        let mut bs = vec![f64::NAN; ldb * 3];
        for k in 0..3 {
            bs[k * ldb..k * ldb + n].copy_from_slice(&b[k * n..(k + 1) * n]);
        }
        s.solve_strided(&mut bs, 3, ldb).unwrap();
        let mut x = vec![0.0; n * 3];
        for k in 0..3 {
            x[k * n..(k + 1) * n].copy_from_slice(&bs[k * ldb..k * ldb + n]);
            assert!(bs[k * ldb + n..(k * ldb + n + 4).min(bs.len())].iter().all(|v| v.is_nan()), "gap written");
        }
        assert!(relres(&a, &x, &b) < 1e-12);
    }
}

#[test]
fn new_values_same_pattern_refactorize_only() {
    let a = fem2d(20, 20, |_| 1.0, 1.0);
    let a2 = fem2d(20, 20, |c| if c % 3 == 0 { 1e4 } else { 1.0 }, 1.0);
    assert_eq!((&a.cp, &a.ri), (&a2.cp, &a2.ri));
    let n = a.n;
    let b = rhs(n, 1);
    let mut s = solver(Kind::Llt);
    assert!(s.update(n, &a.cp, &a.ri, &a.v).unwrap().reanalyzed);
    let mut x1 = b.clone();
    s.solve(&mut x1, 1).unwrap();
    let info = s.update(n, &a2.cp, &a2.ri, &a2.v).unwrap();
    assert!(!info.reanalyzed);
    assert_eq!((info.analyses, info.factorizations), (1, 2));
    let mut x2 = b.clone();
    s.solve(&mut x2, 1).unwrap();
    assert!(relres(&a2, &x2, &b) < 1e-11);
    assert!(relres(&a2, &x1, &b) > 1e-3, "the first solution must not satisfy the second matrix");
}

#[test]
fn changed_pattern_same_size_is_detected() {
    let a = fem2d(15, 15, |_| 1.0, 1.0);
    let n = a.n;
    // P A P^T: same n, same number of entries, other pattern
    let perm: Vec<usize> = (0..n).map(|i| (i * 37) % n).collect();
    let mut inv = vec![0; n];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    let mut t = Vec::new();
    for j in 0..n {
        for p in a.cp[j]..a.cp[j + 1] {
            t.push((inv[a.ri[p]], inv[j], a.v[p]));
        }
    }
    let pa = Csc::from_triplets(n, t);
    assert_eq!(pa.ri.len(), a.ri.len());
    assert_ne!(pa.ri, a.ri);
    let mut s = solver(Kind::Llt);
    s.update(n, &a.cp, &a.ri, &a.v).unwrap();
    assert_eq!(s.factorize(&pa.cp, &pa.ri, &pa.v).unwrap_err(), Error::PatternChanged);
    let mut x = rhs(n, 1);
    assert_eq!(s.solve(&mut x, 1).unwrap_err(), Error::NotFactorized, "no factor after a failure");
    assert!(s.update(n, &pa.cp, &pa.ri, &pa.v).unwrap().reanalyzed);
    let b = rhs(n, 1);
    let mut x = b.clone();
    s.solve(&mut x, 1).unwrap();
    assert!(relres(&pa, &x, &b) < 1e-12);
}

#[test]
fn bad_values_are_refused_and_drop_the_factor() {
    let a = fem2d(12, 12, |_| 1.0, 1.0);
    let n = a.n;
    let mut s = solver(Kind::Llt);
    s.update(n, &a.cp, &a.ri, &a.v).unwrap();
    let mut v = a.v.clone();
    v[5] = f64::NAN;
    assert_eq!(s.update(n, &a.cp, &a.ri, &v).unwrap_err(), Error::NonFiniteInput { position: 5 });
    let mut x = rhs(n, 1);
    assert_eq!(s.solve(&mut x, 1).unwrap_err(), Error::NotFactorized);
    // one off-diagonal entry changed: the triangles disagree
    s.update(n, &a.cp, &a.ri, &a.v).unwrap();
    let mut v = a.v.clone();
    let p = (a.cp[3]..a.cp[4]).find(|&p| a.ri[p] > 3 && a.v[p] != 0.0).unwrap();
    v[p] *= 1.0 + 1e-9;
    assert_eq!(s.update(n, &a.cp, &a.ri, &v).unwrap_err(), Error::NotSymmetric { row: a.ri[p], col: 3 });
    assert_eq!(s.solve(&mut x, 1).unwrap_err(), Error::NotFactorized);
}

#[test]
fn an_entry_without_its_mirror_is_refused() {
    // [[2, 1], [0, 2]] and [[2, 0], [1, 2]]; the position is reported in the
    // caller's matrix, for compressed columns and for compressed rows
    let cases = [
        (false, [0, 1, 3], [0, 0, 1], 0, 1),
        (false, [0, 2, 3], [0, 1, 1], 1, 0),
        (true, [0, 2, 3], [0, 1, 1], 0, 1),
        (true, [0, 1, 3], [0, 0, 1], 1, 0),
    ];
    for (compressed_rows, cp, ri, row, col) in cases {
        let mut s = Solver::new(Options { compressed_rows, ..opts(Kind::Llt) }).unwrap();
        let err = s.update(2, &cp, &ri, &[2.0, 1.0, 2.0]).unwrap_err();
        assert_eq!(err, Error::NotSymmetric { row, col }, "compressed_rows {compressed_rows}");
    }
}

#[test]
fn one_solver_takes_a_larger_matrix_after_a_small_one() {
    // the solve workspace depends on n, the factor and the direction (LU on
    // compressed rows solves with the transpose) as well as on nrhs
    let (small, large) = (fem2d(2, 2, |_| 1.0, 1.0), fem2d(25, 25, |c| 1.0 + (c % 7) as f64, 1.0));
    for (kind, compressed_rows) in [(Kind::Llt, false), (Kind::Ldlt, false), (Kind::Lu, false), (Kind::Lu, true)] {
        let mut s = Solver::new(Options { compressed_rows, ..opts(kind) }).unwrap();
        for (a, k) in [(&small, 1), (&large, 5), (&small, 2), (&large, 1), (&large, 7)] {
            s.update(a.n, &a.cp, &a.ri, &a.v).unwrap();
            let b = rhs(a.n, k);
            let mut x = b.clone();
            s.solve(&mut x, k).unwrap();
            assert!(relres(a, &x, &b) < 1e-12, "{kind:?} rows {compressed_rows} n={} nrhs={k}", a.n);
        }
    }
}

#[test]
fn invalid_patterns_are_refused() {
    let a = fem2d(5, 5, |_| 1.0, 1.0);
    let n = a.n;
    let mut s = solver(Kind::Llt);
    let mut ri = a.ri.clone();
    ri.swap(a.cp[2], a.cp[2] + 1);
    assert!(matches!(s.update(n, &a.cp, &ri, &a.v), Err(Error::InvalidInput(_))));
    // a pointer past the number of entries, and n + 1 overflowing (both panicked)
    assert!(matches!(s.update(2, &[0, 3, 2], &[0, 1], &[1.0, 1.0]), Err(Error::InvalidInput(_))));
    assert!(matches!(s.update(usize::MAX, &[], &[], &[]), Err(Error::InvalidInput(_))));
    // drop the diagonal of column 4
    let mut t = Vec::new();
    for j in 0..n {
        for p in a.cp[j]..a.cp[j + 1] {
            if !(j == 4 && a.ri[p] == 4) {
                t.push((a.ri[p], j, a.v[p]));
            }
        }
    }
    let m = Csc::from_triplets(n, t);
    assert_eq!(s.update(n, &m.cp, &m.ri, &m.v).unwrap_err(), Error::StructurallySingular { column: 4 });
    let mut x = vec![0.0; n];
    assert_eq!(s.solve(&mut x, 1).unwrap_err(), Error::NotFactorized);
}

#[test]
fn not_positive_definite_names_the_column_in_both_kernels() {
    // a definite matrix with one negative diagonal entry, as an isolated node
    let a = fem2d(25, 25, |_| 1.0, 1.0);
    let n = a.n;
    let bad = 311;
    let mut t = Vec::new();
    for j in 0..n {
        for p in a.cp[j]..a.cp[j + 1] {
            let i = a.ri[p];
            if i == bad || j == bad {
                if i == j {
                    t.push((i, j, -1.0));
                } else {
                    t.push((i, j, 0.0));
                }
            } else {
                t.push((i, j, a.v[p]));
            }
        }
    }
    let m = Csc::from_triplets(n, t);
    for nodal in [Nodal::Simplicial, Nodal::Supernodal] {
        let mut s = Solver::new(Options { nodal, ..Options::default() }).unwrap();
        match s.update(n, &m.cp, &m.ri, &m.v) {
            Err(Error::NotPositiveDefinite { column, .. }) => assert_eq!(column, bad, "{nodal:?}"),
            other => panic!("{nodal:?}: {other:?}"),
        }
        assert_eq!(s.info().fail_column, Some(bad));
    }
    // LDL^T factors it and counts the negative pivot
    let mut s = solver(Kind::Ldlt);
    s.update(n, &m.cp, &m.ri, &m.v).unwrap();
    assert_eq!(s.info().neg_pivots, 1);
}

#[test]
fn singular_neumann_matrix_is_refused_by_every_kind() {
    let a = fem2d(20, 20, |c| if c % 5 == 0 { 10.0 } else { 1.0 }, 0.0);
    let n = a.n;
    for kind in [Kind::Llt, Kind::Ldlt, Kind::Lu] {
        let mut s = solver(kind);
        let e = s.update(n, &a.cp, &a.ri, &a.v).unwrap_err();
        assert!(
            matches!(e, Error::NotPositiveDefinite { .. } | Error::NumericallySingular { .. } | Error::ZeroPivot { .. } | Error::Singular { .. }),
            "{kind:?}: {e:?}"
        );
        let mut x = rhs(n, 1);
        assert_eq!(s.solve(&mut x, 1).unwrap_err(), Error::NotFactorized, "{kind:?}");
    }
    // the same matrix with one node grounded is fine
    let g = fem2d(20, 20, |c| if c % 5 == 0 { 10.0 } else { 1.0 }, 1.0);
    let mut s = solver(Kind::Llt);
    s.update(n, &g.cp, &g.ri, &g.v).unwrap();
    // the check is a detection threshold, not a pass condition: it can be
    // switched off, and then the factor is kept as computed
    let mut s = Solver::new(Options { kind: Kind::Ldlt, singular_pivot_ratio: Some(0.0), ..Options::default() }).unwrap();
    let r = s.update(n, &a.cp, &a.ri, &a.v);
    assert!(r.is_ok() || matches!(r, Err(Error::ZeroPivot { .. })));
}

#[test]
fn lu_on_compressed_rows_of_a_nonsymmetric_matrix() {
    // advection-diffusion-like: symmetric stiffness plus a skew part
    let a = fem2d(14, 11, |_| 1.0, 1.0);
    let n = a.n;
    let mut t = Vec::new();
    for j in 0..n {
        for p in a.cp[j]..a.cp[j + 1] {
            let i = a.ri[p];
            let skew = if i + 1 == j { 0.3 } else if j + 1 == i { -0.3 } else { 0.0 };
            t.push((i, j, a.v[p] + skew));
        }
    }
    let m = Csc::from_triplets(n, t);
    let b = rhs(n, 1);
    let xd = dense_solve(m.dense(), b.clone());
    // the same matrix handed as CSR: the CSC arrays of its transpose
    let csr = m.transpose();
    for (arrays, rows) in [(&m, false), (&csr, true)] {
        let mut s = Solver::new(Options { kind: Kind::Lu, compressed_rows: rows, ..Options::default() }).unwrap();
        s.update(n, &arrays.cp, &arrays.ri, &arrays.v).unwrap();
        let mut x = b.clone();
        s.solve(&mut x, 1).unwrap();
        let d = x.iter().zip(&xd).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max);
        assert!(d < 1e-10, "compressed_rows={rows}: {d}");
        assert!(s.info().rcond.unwrap() > 0.0);
    }
}

#[test]
fn rcond_estimate_is_close_to_the_dense_condition_number() {
    let a = fem2d(8, 6, |c| if c % 4 == 0 { 100.0 } else { 1.0 }, 0.1);
    let n = a.n;
    let mut s = solver(Kind::Llt);
    s.update(n, &a.cp, &a.ri, &a.v).unwrap();
    let est = 1.0 / s.rcond_estimate().unwrap();
    // exact 1-norm condition number from the dense inverse
    let d = a.dense();
    let norm1 = (0..n).map(|j| (0..n).map(|i| d[i][j].abs()).sum::<f64>()).fold(0.0, f64::max);
    let mut inv_norm1: f64 = 0.0;
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        let c = dense_solve(d.clone(), e);
        inv_norm1 = inv_norm1.max(c.iter().map(|v| v.abs()).sum());
    }
    let exact = norm1 * inv_norm1;
    assert!(est <= exact * (1.0 + 1e-8) && est >= exact / 3.0, "estimate {est}, exact {exact}");
}

#[test]
fn shared_analysis_is_reused_only_for_the_identical_pattern() {
    rustsolve::clear_shared_analyses();
    let o = Options { share_analysis: true, ..Options::default() };
    // a size no other test uses, so tests running in parallel do not interfere
    let a = fem2d(41, 3, |_| 1.0, 1.0);
    let a2 = fem2d(41, 3, |c| 1.0 + c as f64, 1.0);
    let n = a.n;
    let b = rhs(n, 1);
    let mut s1 = Solver::new(o).unwrap();
    s1.update(n, &a.cp, &a.ri, &a.v).unwrap();
    assert!(!s1.info().analysis_shared);
    let mut s2 = Solver::new(o).unwrap();
    s2.update(n, &a2.cp, &a2.ri, &a2.v).unwrap();
    assert!(s2.info().analysis_shared);
    assert_eq!(s2.info().analyses, 0);
    let mut x = b.clone();
    s2.solve(&mut x, 1).unwrap();
    assert!(relres(&a2, &x, &b) < 1e-12);
    // another pattern of the same size is not taken
    let c = fem2d(3, 41, |_| 1.0, 1.0);
    assert_eq!(c.n, n);
    let mut s3 = Solver::new(o).unwrap();
    s3.update(n, &c.cp, &c.ri, &c.v).unwrap();
    assert!(!s3.info().analysis_shared);
    // LU never takes a Cholesky analysis
    let mut s4 = Solver::new(Options { kind: Kind::Lu, ..o }).unwrap();
    s4.update(n, &a.cp, &a.ri, &a.v).unwrap();
    assert!(!s4.info().analysis_shared);
}

#[test]
fn nested_dissection_needs_its_feature() {
    let r = Solver::new(Options { ordering: Ordering::NestedDissection, ..Options::default() });
    if cfg!(feature = "nd") {
        let a = fem2d(30, 30, |_| 1.0, 1.0);
        let mut s = r.unwrap();
        s.update(a.n, &a.cp, &a.ri, &a.v).unwrap();
        let b = rhs(a.n, 1);
        let mut x = b.clone();
        s.solve(&mut x, 1).unwrap();
        assert!(relres(&a, &x, &b) < 1e-12);
    } else {
        assert!(matches!(r, Err(Error::Unsupported(_))));
    }
}
