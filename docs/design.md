# Design

rustsolve is a general sparse direct solver library: Rust API, C ABI,
Python package. Its first user is pyGIMLi's `LinSolver`, and that use shaped
the rules below. Nothing in the library depends on pyGIMLi.

## Roadmap toward a general library

- Complex matrices: Hermitian LL^H / LDL^H and complex LU (faer supports
  `c64`). This would also cover pyGIMLi's complex resistivity (see Open
  questions).
- `f32`.
- Ordering: AMD, nested dissection, and an automatic choice between them.
- Wheels for Windows, macOS and Linux (maturin, CI); a crates.io release.
- Interop: `sprs` / `nalgebra-sparse` / `faer` matrices on the Rust side.

## First integration: what pgcore needs

pyGIMLi's C++ core (pgcore, libgimli) solves its finite-element systems in
`LinSolver`. Its only backend is `CHOLMODWrapper` (`core/src/cholmodWrapper.cpp`).
In gimli 1.5.0 to 1.5.5, libgimli imports 11 CHOLMOD and 11 UMFPACK functions and
nothing else from SuiteSparse.

- Real symmetric matrices go through `cholmod_analyze` + `cholmod_factorize`.
  This uses CHOLMOD's GPL Supernodal module on larger matrices.
  `cholmod_sdmult` is in the GPL MatrixOps module.
- A full-storage matrix that is not symmetric (absolute tolerance 1e-12),
  `SolverType::UMFPACK`, or `BERTUSEUMFPACK=1` go through `umfpack_di_*`.
- Complex symmetric matrices go through `umfpack_zi_*`.
- Statuses are not checked. pgcore's DC forward only prints "Ooops" when the
  residual exceeds 1e-6.

The ERT forward (`DCMultiElectrodeModelling::calculateK_`,
`DCSRMultiElectrodeModelling::calculateK`) assembles one matrix per 2.5D
wavenumber, builds a new `LinSolver` for it with `setMatrix(S, 1)`, and solves
one right-hand side per electrode. The ERT forward only reaches the real
Cholesky path.

## Interface

C ABI, `crates/rustsolve-capi/include/rustsolve.h`, ABI version 1:

```c
int32_t rs_default_options(rs_options *opt);
int32_t rs_new(const rs_options *opt, rs_solver **out);
void    rs_free(rs_solver *s);
int32_t rs_update_i32(rs_solver *s, int64_t n, const int32_t *col_ptr, const int32_t *row_idx, const double *values);
int32_t rs_update_i64(rs_solver *s, int64_t n, const int64_t *col_ptr, const int64_t *row_idx, const double *values);
int32_t rs_solve(rs_solver *s, int64_t nrhs, double *b, int64_t ldb);
int32_t rs_rcond_estimate(rs_solver *s, double *rcond);
int32_t rs_get_info(const rs_solver *s, rs_info *info);
const char *rs_last_error(const rs_solver *s);
const char *rs_status_name(int32_t status);
void    rs_clear_shared_analyses(void);
```

The Rust API (`crates/rustsolve`) offers the same operations: `Solver::new`,
`analyze`, `factorize`, `update`, `solve`, `solve_strided`, `rcond_estimate`
and `info`.

### Semantics

- **update** analyzes only when the pattern differs from the handle's last
  one, compared entry by entry (pointers and indices), then factorizes.
  `info.reanalyzed` tells which path ran. The lower-level **factorize**
  refuses a pattern that is not the analyzed one (`PATTERN_CHANGED`). This
  matters because faer does not check: it panicked on a mismatched pattern
  in the cases tried, and nothing guarantees that it always would.
- **A failed update or factorization** clears the factor, including an
  input that the C ABI or the Python layer refuses before it reaches the
  factorization. A failed solve (a wrong right-hand side) leaves the factor.
  Each failure has its own status:
  - `NON_FINITE_INPUT`
  - `NOT_SYMMETRIC {row, col}`
  - `STRUCTURALLY_SINGULAR {column}`
  - `NOT_POSITIVE_DEFINITE` and `ZERO_PIVOT`, with the elimination step and
    the original column. faer reports the step 1-based from its simplicial
    kernel and 0-based from its supernodal one; rustsolve normalizes both to
    0-based.
  - `NUMERICALLY_SINGULAR`
  - `SINGULAR`
  - `NON_FINITE_SOLUTION`

  A panic is caught at the C boundary (`RS_PANIC`).
- **Singularity detection** is a detection threshold, not a pass condition,
  and never perturbs the matrix.
  - LL^T / LDL^T: min|pivot| / max|pivot| < n·eps. In the benchmarks the
    smallest ratio on a legitimate system was 1.7e-8 (a 1e7 resistivity
    contrast, cond 2.6e11). A pure-Neumann (singular) matrix gave about
    5e-15.
  - LU: faer does not flag numerically zero pivots, so rustsolve uses a
    Hager–Higham reciprocal condition estimate below eps (a few extra
    solves). The Neumann matrix gave 3.9e-18.
  - Both thresholds can be set, or switched off with 0.
- **check_symmetry** (symmetric kinds, on by default) refuses an
  off-diagonal entry v whose mirror m gives |v − m| > `symmetry_tol` ·
  max(|v|, |m|), and reports its row and column in the caller's matrix. A
  mirror that is not stored counts as zero, so a nonzero entry stored without
  one is refused (for `symmetry_tol` < 1). With the check off, the
  factorization reads only the lower triangle of the column view (the upper
  triangle of CSR arrays). Every stored value is still checked to be
  finite, also in the triangle that is not read, and `rcond_estimate` uses
  the norm of the symmetric matrix the factor stands for.
- **Options** are checked when a solver is made: thresholds must be finite
  and non-negative. For `singular_pivot_ratio` and `min_rcond`, `None`, or a
  finite negative value in the C ABI, selects the default; `symmetry_tol`
  has no default marker.
- **Python input**: matrices and right-hand sides must hold booleans,
  integers or floating-point numbers; complex and object arrays raise
  `TypeError`. A tuple names its format, "csr" or "csc"; its `indptr` and
  `indices` must hold integers and `n` must be an integer (nothing is cast:
  0.9 would become index 0).
- **compressed_rows**: the arrays are CSR (pgcore's storage). The solver
  always solves with the matrix the arrays describe; for LU that means a
  transposed solve.
- **share_analysis**: handles of one process share the analysis of an
  identical pattern (same options, same pattern entry by entry). There is
  an LRU of 2. Numeric factors are never shared. pgcore makes a new solver
  per wavenumber, so this gives it analysis reuse without changing pgcore's
  forward code. With a shared analysis the solution is bit-identical to one
  from the handle's own analysis.
- **Workspace**: the solve buffer is kept between solves and reallocated
  whenever a solve needs more than it holds (a larger matrix, more
  right-hand sides).
- **Threads**: factorization and solves have separate thread counts, both
  sequential by default. In the benchmarks, rayon parallelism was slower
  than one thread at every size except around 100 k-node 3D
  factorizations, and slowest in the solves (8P+12E Intel Core Ultra 7
  265; [benchmarks.md](benchmarks.md)).

## pgcore integration

`pgcore/src/cholmodWrapper.cpp` replaces the CHOLMOD/UMFPACK implementation.
The patch adds no data member and no new virtual function to
`cholmodWrapper.h` or `linSolver.h`, only overrides of virtual functions
SolverWrapper already has, so the class layout is unchanged. Every symbol the
shipped `_pygimli_.pyd` imports is still exported, and the shipped bindings
load and pass the checks on Windows and macOS. A rebuilt libgimli can
therefore sit under the bindings shipped with the same pyGIMLi version; see
[pgcore-verification.md](pgcore-verification.md) for the builds checked.
`pgcore/apply_rustsolve.py` patches a gimli 1.5.x tree: it adds the CMake
option `GIMLI_RUSTSOLVE` (default ON) and drops `find_package(UMFPACK/CHOLMOD
REQUIRED)`. See [pgcore/README.md](../pgcore/README.md).

Symmetry is decided once, in the wrapper, with the original's test: an
absolute difference of 1e-12 between the two stored triangles. A matrix that
fails it goes to LU. A matrix that passes is factorized from one triangle,
with `check_symmetry` off, so pgcore's behaviour is unchanged. rustsolve's own test is relative
(1e-12): left on, it refused small-valued matrices that the original
factorized, such as entries of 1e-5 that differ by 5e-13.

A matrix stored as one triangle (`S.stype()` ≠ 0) is read from that
triangle. pgcore's stype −1 keeps columns j ≥ i of each
row, which is the triangle rustsolve reads from compressed rows; stype 1 is
handed over transposed. LU (UMFPACK's place) builds the full symmetric matrix
from the triangle; UMFPACK factorized the triangle alone.

A matrix that is refused leaves no solver behind. The wrapper drops its handle
when `setMatrix` throws, and the patch makes `LinSolver` delete its previous
solver before it builds the next one, so the next `solve` throws instead of
using the previous matrix. `LinSolver` used to keep (and leak) the previous
solver on every `setMatrix`, and returned zeros from `solve` when it had none.

With topography the DC forward solves the primary potentials on a P2 mesh.
When the forward makes that mesh itself, the patch takes the P2 values at the
forward mesh's nodes (the refined mesh starts with them, in order) instead of
interpolating them. `interpolate()` takes the first cell around a node in
heap-address order, and where the shape functions are 0 and 1 only to rounding
(fused multiply-adds on macOS arm64) the value changed from one process to the
next, by up to 2.8e-16 of the largest value. A primary mesh set by the caller
is still interpolated.

- The patch also gives `LinSolver` and `CHOLMODWrapper` their own versions
  of SolverWrapper's one-argument `setMatrix(const ...&)` overloads. These
  are overrides of existing virtual functions, so the class layout is
  unchanged.
  - Calls through `SolverWrapper*` in C++ reach them.
  - The shipped `_pygimli_.pyd` calls SolverWrapper's inline version itself
    (it throws "not yet implemented" and changes nothing), so from Python
    they take effect only with bindings generated from the patched headers.
- The DC forward owns the `LinSolver` it makes through a `std::unique_ptr`,
  and `LinSolver(RSparseMapMatrix)` frees its cached copy when `setMatrix`
  throws.
- The wrapper takes the storage from the matrix (`S.stype()`) and ignores
  the stype given to its constructor. The original decided between CHOLMOD
  and UMFPACK from `S.stype()` too (a full matrix that fails the symmetry
  test went to UMFPACK), and passed the given stype to CHOLMOD only for a
  matrix that passed: ±1 chose the triangle to read, which for a symmetric
  matrix gives the same factor, and 0 meant the normal equations (above).
- `SolverType::LDL` is refused: LDL is never built with rustsolve, and
  without it `LDLWrapper` solved nothing (the output vector stayed as it
  was, or zeros were returned).
- A matrix without entries is refused before pgcore's `rowIdx()` and
  `vals()` are called, which take `&v[0]` of an empty array.
- pyGIMLi's Python class `pg.solver.LinSolver` keeps its previous core
  solver when `factorize` raises, so its `solve` returns the previous
  matrix's solution. That is in pyGIMLi's Python package, outside this
  patch.

The original passed `stype` from `LinSolver` to CHOLMOD unchanged. With
stype 0, the default of `LinSolver(S)` and `pg.solve`, CHOLMOD factorized
A·Aᵀ and solved the normal equations: relres 1e-13 on a 2D operator, where
rustsolve's Cholesky gives 2e-16. What the normal-equations path also
accepted is the last of the open questions below.

## Open questions

1. Non-symmetric matrices that reach the automatic path. Provisional choice:
   explicit LU with a log message, which keeps pgcore's behaviour. The
   alternative is to refuse them. ERT matrices are exactly symmetric.
2. Ordering for 3D. With AMD only, faer's 3D fill is twice CHOLMOD's; nested
   dissection brings it to parity. Options:
   - (a) feral-metis 0.3.0: pure Rust, MIT; young, one author;
   - (b) a geometric nested dissection on the mesh's node coordinates;
   - (c) AMD first, nested dissection when the fill is large.

   The `nd` feature is off by default until this is decided.
3. Thread policy. Proposal: sequential solves; factorization threads chosen
   by problem size, with thresholds measured on the target machines.
4. Complex matrices (pyGIMLi's complex resistivity, UMFPACK's `zi` path;
   the DC forward does not use them). Either take them over with faer's
   complex LU, or keep refusing them, as the current patch does.
5. Symmetric matrices that are not positive definite. The original solved
   two kinds that rustsolve refuses with `NOT_POSITIVE_DEFINITE`:
   - symmetric indefinite, non-singular. Through `LinSolver(S)`, i.e. stype
     0, which `pg.solve` uses, CHOLMOD factorized A·Aᵀ, so the normal
     equations solved any non-singular matrix, with relres 1e-13. Through
     stype 1, small matrices took CHOLMOD's simplicial LDLᵀ and were solved;
     large ones took the supernodal LLᵀ and gave NaN without an error.
   - singular with a consistent right-hand side (pure Neumann). CHOLMOD
     returned one of the solutions, with relres 1e-15.

   Neither occurs in the ERT forward: its matrices are positive definite,
   and pgcore pins a calibration node when a domain has only Neumann
   boundaries. Options:
   - (a) keep refusing (current);
   - (b) on `NOT_POSITIVE_DEFINITE`, factorize again with LDLᵀ, logged; a
     singular matrix still fails the pivot-ratio test;
   - (c) LU for every symmetric matrix that LLᵀ refuses.

   No option perturbs the matrix or brings back an earlier solution.
