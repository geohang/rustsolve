# Benchmarks against CHOLMOD, UMFPACK and SuperLU

Accuracy, speed and failure behaviour of rustsolve's sparse Cholesky, LDLᵀ
and LU (on faer) against the solvers pyGIMLi uses (CHOLMOD and UMFPACK) and
SciPy's SuperLU. The systems are not random matrices: they come from 2D and
3D DC resistivity models (pyGIMLi's finite-element forward operator) driven
by PyHydroGeophysX's example workflows, plus synthetic 3D stiffness matrices
for scaling.

## Systems

They were captured by setting a Python `SolverWrapper` with
`DCMultiElectrodeModelling::setSolver`. Inside it, pgcore's own CHOLMOD call
is made exactly as pgcore makes it, so the workflows run unchanged: the
captured quick-start inversion still ends at chi2 = 0.3143.

| family | source | systems | n | RHS per matrix |
|---|---|---|---|---|
| line2 | `run_ert_manager_inversion(fielddataline2.dat)`: 2D, topography, SR + H2 mesh; 3 responses × 18 wavenumbers | 20 | 3 785 | 72 |
| e4d | the same on an E4D survey, 112 electrodes | 28 | 5 665 | 112 |
| tl_synth | synthetic one-year time lapse, four days | 8 | 24 212 | 72 |
| contrast | line2 mesh; 1e4 Ω·m host with a 1 Ω·m body, a 1e-3 Ω·m body (contrast 1e7), and a checkerboard | 6 | 3 785 | 72 |
| p2_primary | P2 mesh of the topography primary field | 2 | 3 785 | 72 |
| grid3d | 3D tetrahedral DC forward | 1 | 19 600 | 24 |
| scale3d | 3D jittered tetrahedra, P1 stiffness + Dirichlet (scaling only) | 2 | 57 624 / 97 468 | 25 |
| failing | Neumann (singular), indefinite (derived), NaN and ∞-resistivity models | 5 | 767–19 600 | 2–4 |

Matrix updates: one mesh, new values (wavenumbers, Gauss–Newton iterations,
time steps, contrasts). Cache invalidation: a symmetric permutation PAPᵀ of
a real matrix, which keeps n and nnz but changes the pattern.

## Accuracy (worst system, worst column)

relres = ‖Ax−b‖/‖b‖; diff = ‖x−x_CHOLMOD‖/‖x_CHOLMOD‖.

| family | cond | relres CHOLMOD | relres faer LLT | diff faer | diff UMFPACK | diff SuperLU |
|---|---|---|---|---|---|---|
| line2 | 6e3–1.7e5 | 1.1e-12 | 1.2e-12 | 1.4e-13 | 2.3e-13 | 2.3e-13 |
| e4d | 1.6e4–9e5 | 1.2e-12 | 1.3e-12 | 4.2e-13 | 7.4e-13 | 8.9e-13 |
| tl_synth | 2.7e3–6.3e6 | 3.9e-12 | 4.9e-12 | 7.6e-13 | 2.0e-12 | 1.2e-12 |
| contrast | 6.8e5–2.6e11 | 7.9e-10 | 8.0e-10 | 3.0e-8 | 4.5e-8 | 3.0e-8 |
| p2_primary | 3.6e4–1e5 | 1.5e-13 | 1.6e-13 | 8.2e-14 | 2.9e-13 | 3.5e-14 |
| grid3d | 1.9e7 | 4.9e-15 | 6.0e-15 | 5.9e-12 | 5.6e-12 | 8.9e-12 |
| scale3d | 3e8–7.4e9 | 2.9e-12 | 3.3e-12 | 1.4e-12 | 3.1e-13 | 9.8e-12 |

Through pgcore (the solver swapped with `setSolver`):

| test | response faer | response SuperLU | Jacobian faer | Jacobian SuperLU |
|---|---|---|---|---|
| line2, 1e-3 Ω·m body | 3.7e-9 | 1.0e-8 | 1.4e-7 | 4.5e-7 |
| E4D layered | identical | identical | 1.1e-13 | 1.5e-13 |
| grid3d | identical | identical | – | – |
| quick-start inversion | chi2 0.3143103867024 (CHOLMOD 0.3143103867019) | | | |

Response and Jacobian are compared against CHOLMOD. Jacobian =
max|J−J₀|/max|J₀|. The grid3d Jacobian is refused by pgcore itself for that
electrode layout, with CHOLMOD too. faer's Hager–Higham condition estimate
matches SuperLU + onenormest (3.1e6 against 3.13e6; 7.35e9 against 7.37e9).

## Speed (Intel Core Ultra 7 265, 8P+12E; best of 3; ms)

| system | n | CHOLMOD analyze+factor | faer AMD analyze + factor | faer refactor | faer ND analyze + factor | nnz(L) AMD / ND |
|---|---|---|---|---|---|---|
| line2 | 3 785 | 3.2 | 1.8 + 1.2 | 1.2 | 11 + 1.3 | 70k / 71k |
| e4d | 5 665 | 4.9 | 2.7 + 1.9 | 1.5 | 17 + 2.0 | 107k / 112k |
| tl_synth | 24 212 | 70 | 12 + 13 | 13 | 89 + 16 | 1.10M / 0.70M |
| grid3d | 19 600 | 194 | 18 + 121 | 123 | 122 + 67 | 7.3M / 5.0M |
| scale3d | 57 624 | 843 | 75 + 1 120 | 1 120 | 376 + 423 | 40.7M / 23.4M |
| scale3d | 97 468 | 1 850 | 113 + 4 770 | 4 770 | 678 + 1 740 | 96.9M / 47.9M |

- One 2.5D response of line2 (18 wavenumbers): factorizations take 57.5 ms
  with CHOLMOD as pgcore calls it, and 23.8 ms with faer when the analysis is
  kept.
- Solves: CHOLMOD with one RHS per call took 74.6 ms; faer with all 72 RHS
  as one block took 59.1 ms.
- In 2D the solver is a small part of an inversion: about 0.5 s of the
  20 s quick start.
- Multithreading (CHOLMOD with OpenBLAS threads, faer with rayon):
  - 97 k 3D: CHOLMOD 1.32 s; faer ND factorization 1.13 s with 4 threads
    (plus one 0.68 s analysis).
  - Everywhere else, faer's parallel runs were slower than sequential, and
    the parallel solves were much slower (tl_synth: 29 ms sequential,
    332–549 ms with 4–20 threads).

## Failure behaviour

| system | CHOLMOD (pgcore LinSolver) | faer (rustsolve) |
|---|---|---|
| pure-Neumann stiffness, inconsistent RHS | no error, relres 0.89 | LLT `NOT_POSITIVE_DEFINITE`; LDLT `NUMERICALLY_SINGULAR`; LU rcond 3.9e-18 → `NUMERICALLY_SINGULAR` |
| 3D indefinite (derived) | stderr warning, **all-NaN solution, no error** | LLT `NOT_POSITIVE_DEFINITE`; LU solves it (8e-15) |
| 2D indefinite (derived, 1776 negative eigenvalues) | solved: CHOLMOD's simplicial path is LDLᵀ | LLT refuses; LDLT solves it and reports 1776 negative pivots |
| NaN cell in the model | the NaN never reaches the matrix; pgcore returns finite data | — |
| ∞-resistivity block | pgcore forces the 38 empty rows to Dirichlet | solves |
| same n and nnz, other pattern | — | `PATTERN_CHANGED`; faer without the check: panic |
| NaN value / asymmetric value | — | `NON_FINITE_INPUT` / `NOT_SYMMETRIC (row, col)`; later solve `NOT_FACTORIZED` |

## Dependencies and linking

- The dependency tree is permissive throughout; see THIRD_PARTY_NOTICES.md.
- The C ABI DLL (GNU toolchain, fat LTO) imports only KERNEL32,
  api-ms-win-core-synch, msvcrt and ntdll, and exports only the `rs_*`
  functions.
- libgimli 1.5.0 and 1.5.5 import exactly 11 `cholmod_*` and 11
  `umfpack_*` functions, all called from `cholmodWrapper.cpp`.

Scripts: `bench/` (see its README). The systems are regenerated with
`bench/python/export_cases.py` from PyHydroGeophysX's example data.
