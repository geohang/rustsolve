# pyGIMLi (pgcore) without SuiteSparse

pyGIMLi's C++ core solves its finite-element systems in `LinSolver`, whose
only backend is `CHOLMODWrapper`. That backend calls CHOLMOD and UMFPACK,
including CHOLMOD's GPL modules. These files replace it with rustsolve.

- `src/cholmodWrapper.cpp` replaces `core/src/cholmodWrapper.cpp`. The
  patch gives `cholmodWrapper.h` and `linSolver.h` only overrides of
  SolverWrapper's existing virtual functions (no data member, no new virtual
  function), so the class layout stays the same and every symbol the shipped
  bindings import from libgimli is still exported. Behaviour:
  - Real symmetric matrices → Cholesky, from one triangle. Symmetry is the
    original's test (absolute 1e-12 between the triangles). Full-storage
    matrices that fail it, `SolverType::UMFPACK` or `BERTUSEUMFPACK=1` → LU,
    logged when verbose.
  - Every failure throws a GIMLI error, which Python sees as `RuntimeError`.
    The original ignored CHOLMOD's status.
  - Symmetric matrices that are not positive definite are refused. The
    original's stype-0 path (normal equations) solved them.
  - One rustsolve handle per wrapper, with `share_analysis`: pgcore's DC
    forward builds a new `LinSolver` per wavenumber, and they now share one
    analysis of the mesh's pattern.
  - Complex matrices are refused.
  - `RUSTSOLVE_THREADS` and `RUSTSOLVE_ORDERING=nd` are read for
    experiments.
- `apply_rustsolve.py <gimli tree>` applies that file and the other changes
  to any 1.5.x tree:
  - it adds option `GIMLI_RUSTSOLVE` (default ON);
  - it finds `rustsolve.h` and `rustsolve_capi` under `RUSTSOLVE_PREFIX`;
  - it no longer requires UMFPACK/CHOLMOD;
  - it appends `+rustsolve` to `pg.core.versionStr()`. Configure a tree
    without git history with `-DGIMLI_VERSION=<version>`;
  - `LinSolver` drops its previous solver before it builds the next, so a
    refused matrix leaves none and `solve` throws instead of using the
    previous matrix; the DC forward owns its `LinSolver` through a
    `std::unique_ptr`; `SolverType::LDL`, which solved nothing without LDL,
    is refused;
  - with topography, the DC forward takes the primary potentials at the
    forward mesh's nodes from the P2 mesh it makes for them (those nodes are
    its first nodes) instead of interpolating them. `interpolate()` takes the
    first cell around a node by heap address, and on macOS arm64, where
    `TriangleShape::xyz2rst` is compiled to fused multiply-adds, the shape
    functions at a node are 0 and 1 only to rounding, so the value changed
    from one process to the next. On x86-64 both give the same numbers. A
    primary mesh set by the caller (`setPrimaryMesh`) is still interpolated;
  - each file it changes gets a notice under its licence header, as
    Apache-2.0 asks of modified files.
- `patches/gimli-v1.5.{0,4,5}-rustsolve.patch` are the same changes as
  `git apply` patches. `make_patches.py <gimli clone>` regenerates them from
  the tags.
- `check/` holds a C++ check of the patched `LinSolver` and wrapper.

## Which pgcore

| where | pgcore | toolchain of the original |
|---|---|---|
| conda `-c gimli`, Windows | 1.5.0 (py311_1) | MSYS2 MINGW64, GCC 12.2.0, msvcrt, SuiteSparse 6.0.2 |
| conda `-c gimli`, macOS arm64 | 1.5.4 (np22py311) | clang, conda-forge SuiteSparse |
| PyPI wheel | 1.5.5.post1 | MSYS2 MinGW, SuiteSparse 7.12.1 |

## Build (Windows, MSYS2 MINGW64)

MSYS2 installs per user (no administrator rights): unpack
`msys2-base-x86_64-<date>.sfx.exe` from repo.msys2.org/distrib/x86_64/, then in
the **MINGW64** shell (msvcrt, like the 1.5.0 build; not UCRT64):

```bash
pacman -Syu
pacman -S --needed mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-make \
    mingw-w64-x86_64-boost mingw-w64-x86_64-openblas make patch unzip
```

1. rustsolve: `cargo build --release -p rustsolve-capi` (rustup GNU
   toolchain; add `--features nd` for nested dissection). Lay out a prefix:
   `include/rustsolve.h`, `lib/librustsolve_capi.dll.a` and
   `bin/rustsolve_capi.dll` from the target directory.
2. gimli: export the commit the pgcore you replace was built from
   (`pg.core.versionStr()` names it; the conda 1.5.0 build for Windows
   reports `v1.5.0-1-g5991b6ac`, whose C++ is v1.5.0's), then
   `python pgcore/apply_rustsolve.py <gimli>`.
3. libgimli only, for an existing pgcore of the same version:
   `pgcore/build-libgimli-msys2.sh <gimli> <prefix> <work> <version> <python>`.
   - The script builds Triangle from netlib (checked by sha256), with gimli's
     MinGW patch and `-std=gnu17`.
   - It puts placeholders where gimli's CMake requires the binding generator
     and boost_python. The `gimli` target uses none of them.
   - It configures with `GIMLI_VERSION` and builds `make gimli`.

   Because the class layout is unchanged, the `libgimli.dll` it builds
   replaces the one under the same version's `_pygimli_.pyd`. Put
   `rustsolve_capi.dll` beside it and remove `libcholmod`, `libumfpack`,
   `libamd`, `libcamd`, `libccolamd`, `libcolamd` and `libsuitesparseconfig`.
4. Full pgcore with bindings: gimli's own recipe (`.workflow.sh build`:
   `make` and `make pygimli`, with castxml, pygccxml 2.5.0 and pyplusplus
   1.8.5).

`build-libgimli-macos.sh` does step 3 on macOS arm64 with conda-forge clang,
against a conda environment that holds the pgcore to replace.

C++ runtime. The rebuilt libgimli (GCC 16.2.0) imports 120 symbols from
`libstdc++-6.dll`, 3 from `libgcc_s_seh-1.dll` and 4 from
`libwinpthread-1.dll`. All of them exist in the GCC 12.2 runtime that pgcore
1.5.0 ships, and the checks ran on that runtime. The toolchain's newer
runtime should also work (GCC's runtime runs code built by older compilers),
but it was not tested.

## Checks

```bash
bash pgcore/check/run-check-msys2.sh <patched gimli> <build work dir> <rustsolve prefix>
bash pgcore/check/run-check-macos.sh <patched gimli> <build work dir> <rustsolve prefix> <conda env>
```

`check/linsolver_check.cpp` builds against the patched tree and runs 19
checks: the `setMatrix(const ...&)` overrides through `SolverWrapper*`, a
const matrix left unchanged, complex refusals and the solves after them,
`SolverType::LDL`, and matrices without entries or rows.

A comparison of rebuilt and original pgcore (LinSolver cases, ERT forward
and Jacobian with SuperLU as a third backend, inversions, pyGIMLi's own
tests, on Windows and macOS) is in
[docs/pgcore-verification.md](../docs/pgcore-verification.md).

## Known limitations

- Complex matrices (pyGIMLi's complex resistivity) are refused.
- Symmetric indefinite and singular matrices are refused; the original
  solved them through the normal equations.
- With the shipped bindings, `LinSolver.setMatrix` with one argument and a
  complex `setMatrix` on a `CHOLMODWrapper` still raise "not yet
  implemented" inside the bindings and keep the previous matrix; the
  overrides take effect only with bindings generated from the patched
  headers.
- pyGIMLi's Python class `pg.solver.LinSolver` keeps its previous core
  solver when `factorize` raises, so its `solve` returns the previous
  matrix's solution. That is in pyGIMLi's Python package, outside this
  patch.
- Only libgimli has been rebuilt and checked; a full build with regenerated
  bindings has not.
