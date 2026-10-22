# Checking a rebuilt pgcore

A libgimli rebuilt from the gimli sources with `pgcore/apply_rustsolve.py`
was placed under the Python bindings of the same pyGIMLi release and
compared with the original library (CHOLMOD and UMFPACK, SuiteSparse 6.0.2).
Each comparison ran twice in the same Python environment, once with each
library, on the same inputs: meshes and synthetic surveys were made once and
read by both runs.

| | Windows x86-64 | macOS arm64 |
|---|---|---|
| pgcore | 1.5.0 from the gimli conda channel (`libgimli-v1.5.0-1-g5991b6ac`) | `pgcore-1.5.4-np22py311h3a9382f_0` from the gimli channel |
| rebuilt libgimli | GCC 16.2.0, MSYS2 MINGW64, `pgcore/build-libgimli-msys2.sh` | conda-forge clang 21, macOS 13.0 or later, `pgcore/build-libgimli-macos.sh` |
| in the folder | `libgimli.dll` replaced, `rustsolve_capi.dll` added, the seven SuiteSparse DLLs removed | the rebuilt libraries first on `DYLD_LIBRARY_PATH`; every other file the original |
| Python | 3.11, pyGIMLi 1.5.5, NumPy 1.26.4 | the same |

On Windows the rebuilt libgimli kept pgcore's GCC 12.2 runtime DLLs.
rustsolve was built with the `nd` feature. Times are single runs on one
desktop machine (Intel Core Ultra 7 265) and are indicative only.

The Windows results hold with and without the P2 node-value change
(`DCFEM_P2_*` in `apply_rustsolve.py`): both builds give bit-identical
arrays. The macOS results come from a build without it; its expected effect
there is described at the end.

## Binaries

- The rebuilt libgimli no longer imports `libcholmod` or `libumfpack`;
  `rustsolve_capi` imports system libraries only.
- Every symbol the shipped bindings import from libgimli (1140 on Windows,
  1196 on macOS) is exported by the rebuilt library, and no export of the
  original is missing. Its own imports exist in the C++ runtime, OpenMP and
  OpenBLAS libraries pgcore ships.
- No SuiteSparse module is loaded in the rebuilt process.
  `pg.core.versionStr()` ends in `+rustsolve` and `LinSolver.solverName()`
  reports `rustsolve`.
- UMFPACK and CHOLMOD's GPL modules (Supernodal, Modify, MatrixOps) are in
  no rebuilt binary.

## LinSolver

Each case goes through `pg.core.LinSolver`, with `LinSolver(S)` or
`setMatrix(S, stype)`; relres is ‖Ax−b‖/‖b‖. The SPD matrix is a 2D
five-point operator (n = 1600, coefficients over four decades); the others
are derived from it or are standard test matrices: shifted between two
eigenvalues (indefinite), entries of about 1e-5 with one changed by 5e-13
(near-symmetric), upwind convection–diffusion (non-symmetric), and a
pure-Neumann graph Laplacian with a consistent right-hand side (singular).

| case | original | rebuilt | relres original | relres rebuilt |
|---|---|---|---|---|
| SPD, `LinSolver(S)` (stype 0) | CHOLMOD | Cholesky | 1.3e-13 | 1.9e-16 |
| SPD, stype 1 / −1 | CHOLMOD | Cholesky | 1.7e-16 / 1.6e-16 | 1.9e-16 |
| SPD, new values, same solver | | | 2.2e-16 | 2.3e-16 |
| near-symmetric, stype 0 / 1 | CHOLMOD | Cholesky | 1.8e-10 / 1.8e-13 | 1.1e-12 |
| non-symmetric | UMFPACK | LU | 1.3e-16 | 1.9e-16 |
| symmetric indefinite, stype 0 / 1 | solved | `NOT_POSITIVE_DEFINITE` | 1.6e-13 / 1.5e-16 | — |
| singular, stype 0 / 1 | one solution | `NOT_POSITIVE_DEFINITE` | 4.3e-15 / 2.1e-16 | — |
| SPD stored as one triangle, stype −1 / 1 | CHOLMOD | Cholesky | 1.6e-16 / 1.7e-16 | 1.9e-16 |
| SPD stored as one triangle, LU | UMFPACK on the triangle alone | LU | 2.2e-1 | 2.2e-16 |
| SPD, then −A on the same solver, then solve | −A solved | `setMatrix` and `solve` raise | 1.7e-16 | — |

- Stype 0 is the default of `LinSolver(S)` and `pg.solve`. There the
  original factorized A·Aᵀ and solved the normal equations, hence its
  1.3e-13, and also why it solved the indefinite and singular matrices.
- UMFPACK factorized a matrix stored as one triangle as if the triangle
  were the whole matrix; the rebuilt core builds the full symmetric matrix
  first.
- After a refused matrix the rebuilt core keeps no solver, so the next
  `solve` raises instead of returning the previous matrix's solution.
- Two calls behave the same in both builds because the shipped bindings
  never reach libgimli: `setMatrix` with one argument, and a complex
  `setMatrix` on a `CHOLMODWrapper`. Both raise "not yet implemented" in
  SolverWrapper's inline code and leave the previous matrix in place.
- pyGIMLi's Python class `pg.solver.LinSolver` keeps its previous core
  solver when `factorize` raises: the original solves −A, the rebuilt core
  refuses it and then solves with A.

`pgcore/check/linsolver_check.cpp` covers the C++ side (19 checks): the
`setMatrix(const ...&)` overrides through `SolverWrapper*`, a const matrix
left unchanged, complex refusals and the solves after them, `SolverType.LDL`,
and matrices without entries or rows. It passes on both platforms. On macOS,
libgimli is linked with `-undefined dynamic_lookup` and its static
initializer calls `Py_IsInitialized()`, so `run-check-macos.sh` links
libpython (without starting Python).

## ERT forward and Jacobian (Windows)

Response: largest relative difference. Jacobian: largest absolute
difference over max |J|. C is the original, R the rebuilt core, S SciPy's
SuperLU for every solve (primary potentials included, through pgcore's
solver hook); S_C and S_R are S run on each library.

| case | data | response R–C | S–C | S–R | S_C–S_R | Jacobian R–C | S–C | S–R | t response C / R (s) | t Jacobian C / R (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2D line, 1e7 contrast | 936 | 6.5e-07 | 3.0e-07 | 3.6e-07 | 0 | 3.7e-05 | 2.6e-05 | 1.6e-05 | 4.32 / 3.62 | 0.20 / 0.20 |
| 2D line, homogeneous | 936 | 0 | 0 | 0 | 0 | 5.2e-14 | 6.9e-14 | 5.3e-14 | 4.78 / 3.74 | 0.20 / 0.19 |
| E4D survey, layered | 3647 | 0 | 0 | 0 | 0 | 1.9e-13 | 2.3e-13 | 1.6e-13 | 10.09 / 10.11 | 2.24 / 1.45 |
| 3D grid, one body | 22 | 0 | | | | | | | 13.50 / 22.27 | |

- With SuperLU for every solve the two libraries give bit-identical
  responses (S_C–S_R = 0), so the assembly is unchanged.
- At a 1e7 contrast the three backends agree to within 7e-7, and which pair
  is closest changes between runs because the mesh is rebuilt each time
  (Triangle moves nodes by about 1e-13). The primary potentials, solved on
  a P2 mesh, agree to 2–6e-13 between backends; the secondary right-hand
  side (S₁/ρ − S)·u_p multiplies that by the contrast. With both backends
  on the same primary potentials R–C falls to 7e-10. The smallest-wavenumber
  matrix has cond₂ = 1.75e11; rustsolve, SuperLU and dense LU are each
  within 4e-8 of dense Cholesky on it.
- A zero response difference does not mean an unused solver: the
  Jacobians differ by 1e-13. At the electrodes the analytic primary
  potential dominates and the solvers' rounding stays below one ulp.
- 3D grid: 1.6–2× slower and 3.39 GB peak memory against 2.59 GB. With AMD
  ordering faer's fill in 3D is about twice CHOLMOD's; nested dissection
  (`nd`) brings it to parity ([benchmarks.md](benchmarks.md)).

## Inversions (Windows)

| | original | rebuilt |
|---|---|---|
| 2D line, 4 iterations: chi² | 0.31431038670191935 | 0.31431038670192396 |
| 2D line: model | | max relative difference 3.7e-13 |
| 2D line: wall time | 10.2 s | 9.2 s |
| synthetic survey, 3 iterations: chi² per iteration | 167.1058266749, 4.92676682272, 0.35770363639 | identical |
| synthetic survey: model | | 3.4e-15 |
| time-lapse, 3 surveys, 2 iterations: chi² | 9 values | agree to 15 digits |
| time-lapse: models | | 1.1e-15 |

The 2D line inversion builds its mesh in every process, so the same library
run twice differs by up to 6e-10 in chi²; the difference between the
libraries is within that spread.

## pyGIMLi's test suite

`pygimli/testing` ran one file per pytest process (300 s limit per file), in
pytest's importlib mode so that each run imports the core under test; each
process recorded the core it loaded.

| | passed | failed | timed out |
|---|---|---|---|
| Windows, original | 83 | 2 | 3 files |
| Windows, rebuilt | 82 | 4 | 3 files |
| macOS, original | 82 | 3 | 3 files |
| macOS, rebuilt | 81 | 5 | 3 files |

The two differences come from complex matrices, which the rebuilt core
refuses: `test_SparseMatrix::test_ComplexMatrix`, and the module-level
complex-resistivity simulation in `test_ERTManager_simulate.py`. Everything
else has the same outcome on both sides. The failures common to both need
TetGen, which the environment does not have (and, on macOS,
`test_PhysicsManagers::test_ERT`); the same three files time out on both
sides.

## With rustmesh meshes (Windows)

The same forward and inversion comparison on meshes from
[rustmesh](https://github.com/geohang/rustmesh) (2D, smallest angle 28°)
instead of Triangle:

| survey | rustmesh cells (inverted) | Triangle cells (inverted) |
|---|---|---|
| 2D line | 3868 (3230) | 1794 (1356) |
| E4D survey | 7613 (6982) | 2694 (2185) |
| synthetic series | 1511 (1019) | 892 (427) |

- At a 1e7 contrast the three backends agree to within 4.4e-7; with the
  same solver the two libraries again give identical responses. The
  homogeneous and E4D responses are identical, the Jacobians within 1e-12.
- The 2D line inversion on the fixed rustmesh mesh: chi² differs by 2.7e-10
  and the model by 3.2e-9, solver rounding carried through four
  Gauss–Newton iterations, far below a 3 % data error.
- The synthetic and time-lapse inversions agree to 15 digits.
- rustsolve was 16–24 % faster in the response and 20–23 % in the Jacobian
  on these meshes.

## macOS arm64

- The C++ check, the binary checks and the LinSolver cases give the same
  outcomes as on Windows.
- Forward, 1e7 contrast: response R–C 2.6e-7, S–C 5.4e-7, S–R 2.8e-7;
  Jacobian 1.2e-5, 2.9e-5, 2.6e-5. S_C–S_R = 0 on every case. Homogeneous
  line and E4D: the responses agree to 4e-10, the Jacobians to 1.6e-13.
- 3D grid: identical responses; rustsolve takes 22.5 s against 10.9 s and,
  with AMD ordering, more than 4 GB.
- Inversions: the 2D line's chi² is the same on both sides and the model
  within 4e-14; the synthetic and time-lapse inversions agree to 15 digits.

### Primary potentials between processes

Run again in new processes on the same inputs, the rebuilt core's 1e7
response on macOS moved by 5e-13 to 5e-8 and its Jacobian by up to 2.3e-6
of max |J| (three repeats), while its solves on fixed primary potentials
repeated bit for bit.
The cause is in pgcore, not in the solver:

- With topography pgcore solves the primary potentials on a P2 mesh and
  interpolates them to the forward mesh's nodes. `findCell` takes the first
  cell around a node in `std::set<Cell*>` order, that is by heap address,
  so the cell used changes from one process to the next.
- clang compiles `TriangleShape::xyz2rst` to fused multiply-adds, so at a
  cell vertex the shape functions are 0 and 1 only to rounding. The
  interpolated value then differs from the node value by up to 2.8e-16 of
  the largest value, depending on the cell taken. GCC on x86-64 uses
  separate multiplies and subtractions, and the interpolation is exact at
  the nodes.
- Emulating the fused arithmetic exactly on Windows reproduces the macOS
  differences (1997 against 2129 of 3785 nodes, both at 2.7997e-16).

The patch therefore takes the P2 values at the forward mesh's nodes, which
the refined mesh holds first and in order (positions are checked), instead
of interpolating them; a primary mesh set by the caller is still
interpolated. On Windows nothing changes, bit for bit. On macOS the rebuilt
core's primary potentials are expected to change by up to 2.8e-16 of the
largest value at about 2000 of 3785 nodes, and the 1e7 responses by about
1.8e-7, the gap measured between node values and interpolation in one
process. That is expected, not measured: no macOS build with the change has
been run.

## Not covered

- A full pgcore build with regenerated bindings.
- The newer GCC runtime libraries on Windows.
- A macOS build with the P2 node-value change.
- Complex matrices, which the rebuilt core refuses.
- rustmesh's 3D meshes.
- Why the original hangs on macOS when OpenBLAS and OpenMP are limited to
  one thread.
