# Changelog

## [0.1.0] - 2026-10-22

First public version.

### Added
- `rustsolve`: Cholesky, LDLᵀ and LU on faer 0.24.4 behind analyze /
  factorize / update / solve, with exact pattern checks, explicit errors
  (the failing pivot and its original column), singularity detection (pivot
  ratio, reciprocal condition estimate), compressed rows or columns,
  analyses shared between solvers, separate thread counts for factorization
  and solves, and optional nested dissection (`nd`).
- `rustsolve-capi`: C ABI, version 1 (`include/rustsolve.h`).
- `rustsolve-py`: Python package that takes `scipy.sparse` input and raises
  typed exceptions.
- `pgcore/`: a replacement `cholmodWrapper.cpp` and patches that move
  pyGIMLi 1.5.0, 1.5.4 and 1.5.5 from CHOLMOD/UMFPACK to rustsolve, build
  scripts for libgimli on Windows (MSYS2) and macOS arm64, and a C++ check.
- `bench/`: export of ERT systems through pyGIMLi, reference runs with
  CHOLMOD, UMFPACK and SuperLU, and comparison tables
  ([docs/benchmarks.md](docs/benchmarks.md)).
