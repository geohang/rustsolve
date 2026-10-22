//! Python bindings: `rustsolve._rustsolve.Factor`. The friendly layer
//! (scipy.sparse input, exception classes) is python/rustsolve/__init__.py.

use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustsolve::{Error, Kind, Nodal, Options, Ordering, Solver};

pyo3::create_exception!(_rustsolve, SolverError, PyRuntimeError);

fn err(e: Error, s: &Solver) -> PyErr {
    let i = s.info();
    SolverError::new_err((e.name(), e.to_string(), i.fail_step, i.fail_column))
}

fn indices(a: &PyReadonlyArray1<'_, i64>, what: &str) -> PyResult<Vec<usize>> {
    a.as_array()
        .iter()
        .map(|&v| usize::try_from(v).map_err(|_| PyValueError::new_err(format!("negative entry in {what}"))))
        .collect()
}

/// A factorization of one sparse matrix at a time.
#[pyclass(module = "rustsolve._rustsolve")]
struct Factor {
    s: Solver,
}

#[pymethods]
impl Factor {
    #[new]
    #[pyo3(signature = (kind="cholesky", ordering="amd", nodal="auto", threads=1, solve_threads=1,
                        check_symmetry=true, symmetry_tol=1e-12, singular_pivot_ratio=None,
                        min_rcond=None, compressed_rows=false, share_analysis=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        kind: &str,
        ordering: &str,
        nodal: &str,
        threads: usize,
        solve_threads: usize,
        check_symmetry: bool,
        symmetry_tol: f64,
        singular_pivot_ratio: Option<f64>,
        min_rcond: Option<f64>,
        compressed_rows: bool,
        share_analysis: bool,
    ) -> PyResult<Self> {
        let kind = match kind {
            "cholesky" | "llt" => Kind::Llt,
            "ldlt" => Kind::Ldlt,
            "lu" => Kind::Lu,
            k => return Err(PyValueError::new_err(format!("kind must be 'cholesky', 'ldlt' or 'lu', not {k:?}"))),
        };
        let ordering = match ordering {
            "amd" => Ordering::Amd,
            "nd" | "nested_dissection" => Ordering::NestedDissection,
            o => return Err(PyValueError::new_err(format!("ordering must be 'amd' or 'nd', not {o:?}"))),
        };
        let nodal = match nodal {
            "auto" => Nodal::Auto,
            "simplicial" => Nodal::Simplicial,
            "supernodal" => Nodal::Supernodal,
            v => return Err(PyValueError::new_err(format!("nodal must be 'auto', 'simplicial' or 'supernodal', not {v:?}"))),
        };
        let opts = Options {
            kind,
            ordering,
            nodal,
            check_symmetry,
            symmetry_tol,
            compressed_rows,
            singular_pivot_ratio,
            min_rcond,
            factor_threads: threads,
            solve_threads,
            share_analysis,
        };
        let s = Solver::new(opts).map_err(|e| SolverError::new_err((e.name(), e.to_string(), None::<usize>, None::<usize>)))?;
        Ok(Factor { s })
    }

    /// Analyze (when the pattern is new) and factorize.
    fn update(
        &mut self,
        py: Python<'_>,
        n: &Bound<'_, PyAny>,
        indptr: &Bound<'_, PyAny>,
        indices_: &Bound<'_, PyAny>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        // an input rejected below must not leave the previous factor in place;
        // the arguments are converted here, after it is dropped, not by PyO3
        // before the method runs
        self.s.drop_factor();
        let n: usize = n.extract()?;
        let indptr: PyReadonlyArray1<'_, i64> = indptr.extract()?;
        let indices_: PyReadonlyArray1<'_, i64> = indices_.extract()?;
        let data: PyReadonlyArray1<'_, f64> = data.extract()?;
        let cp = indices(&indptr, "indptr")?;
        let ri = indices(&indices_, "indices")?;
        let v: Vec<f64> = data.as_array().iter().copied().collect();
        let s = &mut self.s;
        let r = py.detach(|| s.update(n, &cp, &ri, &v).map(|_| ()));
        r.map_err(|e| err(e, &self.s))
    }

    /// Solve A x = b for b of shape (n,) or (n, k); returns x with b's shape.
    fn solve<'py>(&mut self, py: Python<'py>, b: PyReadonlyArrayDyn<'py, f64>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let n = self.s.info().n;
        let shape = b.shape().to_vec();
        let (rows, k) = match shape.as_slice() {
            [r] => (*r, 1),
            [r, c] => (*r, *c),
            _ => return Err(PyValueError::new_err("b must have one or two dimensions")),
        };
        if rows != n {
            return Err(PyValueError::new_err(format!("b has {rows} rows, the matrix {n}")));
        }
        let a = b.as_array();
        // column-major work array
        let mut x = vec![0.0f64; n * k];
        for (idx, v) in a.indexed_iter() {
            let (i, j) = if shape.len() == 1 { (idx[0], 0) } else { (idx[0], idx[1]) };
            x[j * n + i] = *v;
        }
        let s = &mut self.s;
        let r = py.detach(|| s.solve(&mut x, k));
        r.map_err(|e| err(e, &self.s))?;
        let mut out = ArrayD::<f64>::zeros(IxDyn(&shape));
        for (idx, v) in out.indexed_iter_mut() {
            let (i, j) = if shape.len() == 1 { (idx[0], 0) } else { (idx[0], idx[1]) };
            *v = x[j * n + i];
        }
        Ok(out.into_pyarray(py))
    }

    /// Reciprocal 1-norm condition estimate with the current factor.
    fn rcond(&mut self, py: Python<'_>) -> PyResult<f64> {
        let s = &mut self.s;
        let r = py.detach(|| s.rcond_estimate());
        r.map_err(|e| err(e, &self.s))
    }

    /// What the last analysis and factorization did.
    #[getter]
    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let i = self.s.info();
        let d = PyDict::new(py);
        d.set_item("n", i.n)?;
        d.set_item("nnz_a", i.nnz_a)?;
        d.set_item("nnz_factor", i.nnz_factor)?;
        d.set_item("supernodal", i.supernodal)?;
        d.set_item("reanalyzed", i.reanalyzed)?;
        d.set_item("analysis_shared", i.analysis_shared)?;
        d.set_item("analyses", i.analyses)?;
        d.set_item("factorizations", i.factorizations)?;
        d.set_item("neg_pivots", i.neg_pivots)?;
        d.set_item("min_pivot", i.min_pivot)?;
        d.set_item("max_pivot", i.max_pivot)?;
        d.set_item("pivot_ratio", i.pivot_ratio)?;
        d.set_item("rcond", i.rcond)?;
        d.set_item("fail_step", i.fail_step)?;
        d.set_item("fail_column", i.fail_column)?;
        Ok(d)
    }
}

#[pyfunction]
fn clear_shared_analyses() {
    rustsolve::clear_shared_analyses();
}

#[pymodule]
fn _rustsolve(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("SolverError", m.py().get_type::<SolverError>())?;
    m.add("has_nested_dissection", cfg!(feature = "nd"))?;
    m.add_class::<Factor>()?;
    m.add_function(wrap_pyfunction!(clear_shared_analyses, m)?)?;
    Ok(())
}
