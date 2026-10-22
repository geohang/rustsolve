# rustsolve

Sparse direct solvers — Cholesky (LL^T), LDL^T and LU — with a Rust core built on
[faer](https://codeberg.org/sarah-quinones/faer), for Rust, C/C++ and Python.
Permissively licensed (MIT OR Apache-2.0), with no SuiteSparse and no GPL code.

The interface is small and strict, written for finite-element and
finite-volume codes that solve many systems on one mesh:

- **analyze → factorize → solve**: an analysis is kept while the sparsity
  pattern stays the same, so a new matrix on the same mesh costs only a
  numeric factorization. Patterns are compared entry by entry.
- **many right-hand sides** in one block solve.
- **explicit failures**: "not positive definite" names the pivot and the
  original column; there are separate errors for singular matrices,
  non-finite input, asymmetric input and a changed pattern. Nothing is
  perturbed to make a factorization succeed, and a failed factorization
  leaves no factor behind that a later solve could use.
- **diagnostics**: pivot range, inertia (LDL^T), a Hager–Higham condition
  estimate.
- **compressed rows or columns**, 32- or 64-bit indices, an analysis shared
  between solvers of one process, separate thread counts for factorization
  and solves.

The library has no ties to any one application. One worked example ships
with it: [pgcore/](pgcore/README.md) replaces the CHOLMOD/UMFPACK backend of
[pyGIMLi](https://www.pygimli.org)'s `LinSolver`, so that a pyGIMLi core can
be built and distributed without SuiteSparse's GPL modules
([PyHydroGeophysX](https://github.com/geohang/PyHydroGeophysX) uses it this
way). [rustmesh](https://github.com/geohang/rustmesh) is a companion mesher
written in the same spirit.

Status: pre-alpha. Real `f64` matrices only; complex matrices are on the roadmap.

## Python

```python
import numpy as np, scipy.sparse as sp, rustsolve

A = sp.diags([-1.0, 2.1, -1.0], [-1, 0, 1], shape=(1000, 1000), format="csr")
f = rustsolve.factorize(A)                 # "cholesky" (default), "ldlt" or "lu"
x = f.solve(np.ones(1000))                 # b of shape (n,) or (n, k)
f.update(A * 3.0)                          # same pattern: numeric factorization only
print(f.info["reanalyzed"], 1 / f.rcond())

try:
    rustsolve.factorize(-A)
except rustsolve.NotPositiveDefiniteError as e:
    print(e.column)                        # where it failed
```

Accepts any `scipy.sparse` matrix or array; CSR and CSC are used without
conversion. Exceptions derive from `rustsolve.SolverError`;
`NotPositiveDefiniteError` and `SingularMatrixError` are also
`numpy.linalg.LinAlgError`.

## Rust

```rust
use rustsolve::{Options, Solver};

let mut s = Solver::new(Options::default())?;      // Cholesky, AMD ordering
s.update(n, &col_ptr, &row_idx, &values)?;         // analyze if new, factorize
s.solve(&mut b, nrhs)?;                            // b: n x nrhs, column-major
println!("{:?}", s.info());
```

## C / C++

`crates/rustsolve-capi` builds `rustsolve_capi` as a shared and a static
library; the header is [`include/rustsolve.h`](crates/rustsolve-capi/include/rustsolve.h).

```c
rs_options o; rs_default_options(&o);
rs_solver *s; rs_new(&o, &s);
if (rs_update_i32(s, n, col_ptr, row_idx, values) != RS_OK)
    fprintf(stderr, "%s\n", rs_last_error(s));
rs_solve(s, nrhs, b, n);
rs_free(s);
```

The DLL imports only system libraries (KERNEL32, ntdll, the CRT).

## Accuracy and speed

Measured on 72 systems exported from 2D and 3D DC resistivity forward and
inversion runs, against CHOLMOD, UMFPACK and SuperLU
([docs/benchmarks.md](docs/benchmarks.md)). In short:

- Residuals are the same as CHOLMOD's. The difference from CHOLMOD is
  within the spread between the reference solvers, even at condition
  numbers near 1e11.
- On 2D meshes and medium 3D meshes, rustsolve is as fast as or faster than
  CHOLMOD, and about 2.4× faster when the analysis is reused.
- On about 100 k-node 3D meshes, nested dissection (`nd` feature) is needed
  to match CHOLMOD.

## Layout

| path | what |
|---|---|
| `crates/rustsolve` | the library |
| `crates/rustsolve-capi` | C ABI, `include/rustsolve.h` |
| `crates/rustsolve-py`, `python/` | Python package (PyO3, maturin) |
| `pgcore/` | pyGIMLi example: replacement `cholmodWrapper.cpp` and patches (Apache-2.0), libgimli build scripts, C++ check |
| `bench/` | export of finite-element systems, reference runs, faerbench, comparison |
| `docs/` | design, benchmarks, pgcore results |

## Build

```bash
cargo build --release                 # library + C ABI
cargo test --release -p rustsolve -p rustsolve-capi
cargo build --release -p faerbench    # bench tool
maturin develop --release             # Python package (or scripts/build-python-windows-gnu.ps1)
python -m pytest python/tests
```

Rust 1.84 or newer; developed with 1.99.0 on `x86_64-pc-windows-gnu`.
faer is pinned to `=0.24.4` and `Cargo.lock` is committed. The `nd`
feature (feral-metis, pure Rust) adds nested dissection.

## Licence

MIT OR Apache-2.0, at your option. `pgcore/` holds files derived from
pyGIMLi, which are under Apache-2.0. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
