/*
 * rustsolve.h - C ABI of rustsolve (sparse LL^T, LDL^T and LU on faer).
 *
 * Copyright (c) 2026 Hang Chen
 * SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * Every function returns a status (RS_OK = 0) unless noted. A failed update,
 * including one whose input is rejected before any factorization, leaves the
 * handle without a factor: a later rs_solve returns
 * RS_NOT_FACTORIZED and never a solution of an earlier matrix. Nothing is
 * perturbed to make a factorization succeed. A handle is not thread-safe;
 * separate handles may be used from separate threads.
 */
#ifndef RUSTSOLVE_H
#define RUSTSOLVE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RS_ABI_VERSION 1

/* status codes */
#define RS_OK                     0
#define RS_INVALID_INPUT          1
#define RS_STRUCTURALLY_SINGULAR  2  /* symmetric kind: a column has no diagonal entry */
#define RS_NOT_SYMMETRIC          3  /* stored triangles differ (check_symmetry) */
#define RS_NON_FINITE_INPUT       4
#define RS_PATTERN_CHANGED        5
#define RS_NOT_ANALYZED           6
#define RS_NOT_FACTORIZED         7
#define RS_NOT_POSITIVE_DEFINITE  8  /* LL^T: pivot <= 0, see rs_info.fail_* */
#define RS_ZERO_PIVOT             9  /* LDL^T */
#define RS_NUMERICALLY_SINGULAR  10  /* pivot ratio or rcond below threshold */
#define RS_SINGULAR              11  /* LU: no pivot candidate */
#define RS_NON_FINITE_SOLUTION   12
#define RS_OUT_OF_MEMORY         13
#define RS_INTERNAL              14
#define RS_PANIC                 15  /* caught at the boundary */
#define RS_UNSUPPORTED           16  /* e.g. nested dissection without the nd feature */

/* rs_options.kind */
#define RS_KIND_LLT   0
#define RS_KIND_LDLT  1
#define RS_KIND_LU    2

/* rs_options.ordering (symmetric kinds; LU always uses COLAMD) */
#define RS_ORDERING_AMD  0
#define RS_ORDERING_ND   1

/* rs_options.nodal */
#define RS_NODAL_AUTO        0
#define RS_NODAL_SIMPLICIAL  1
#define RS_NODAL_SUPERNODAL  2

typedef struct rs_solver rs_solver;

/* Fill with rs_default_options, then change fields. 56 bytes. */
typedef struct rs_options {
    int32_t kind;             /* RS_KIND_*; default LLT */
    int32_t ordering;         /* RS_ORDERING_*; default AMD */
    int32_t nodal;            /* RS_NODAL_*; default AUTO */
    int32_t threads;          /* factorization threads: 1 sequential (default), 0 all cores */
    int32_t solve_threads;    /* solve threads: 1 sequential (default), 0 all cores */
    int32_t check_symmetry;   /* symmetric kinds: refuse an entry v whose mirror m gives
                                 |v - m| > symmetry_tol * max(|v|, |m|); a mirror not
                                 stored counts as zero (default 1). With 0 the factor
                                 reads only the lower triangle of CSC (upper of CSR);
                                 all values are still checked to be finite */
    int32_t compressed_rows;  /* arrays are CSR, not CSC (default 0); the solver always
                                 solves with the matrix the arrays describe */
    int32_t share_analysis;   /* reuse the analysis of an identical pattern made by another
                                 handle of this process (default 0); factors are never shared */
    double  symmetry_tol;     /* relative, finite, >= 0 (default 1e-12) */
    double  singular_pivot_ratio; /* LL^T/LDL^T: refuse min|pivot|/max|pivot| below this;
                                     < 0: n * eps (default), 0: no check;
                                     NaN or infinite: RS_INVALID_INPUT */
    double  min_rcond;        /* LU: refuse a reciprocal condition estimate below this;
                                 < 0: eps (default), 0: no check;
                                 NaN or infinite: RS_INVALID_INPUT */
} rs_options;

/* 112 bytes */
typedef struct rs_info {
    int64_t n;
    int64_t nnz_a;
    int64_t nnz_factor;       /* stored entries of L (LL^T/LDL^T), 0 for LU */
    int32_t supernodal;
    int32_t reanalyzed;       /* the last update analyzed the pattern */
    int32_t analysis_shared;  /* the analysis came from another handle (share_analysis) */
    int32_t reserved;
    int64_t analyses;
    int64_t factorizations;
    int64_t neg_pivots;       /* LDL^T */
    double  min_pivot;        /* L_jj^2 (LL^T) or D_jj (LDL^T) */
    double  max_pivot;
    double  pivot_ratio;      /* min|pivot| / max|pivot| */
    double  rcond;            /* LU with the rcond check, else NaN */
    int64_t fail_step;        /* failing elimination step, -1 if none */
    int64_t fail_column;      /* original column at that step, -1 if none */
} rs_info;

uint32_t rs_abi_version(void);  /* returns RS_ABI_VERSION of the library */

int32_t rs_default_options(rs_options *opt);
int32_t rs_new(const rs_options *opt /* NULL: defaults */, rs_solver **out);
void    rs_free(rs_solver *s);

/* Analyze (only when the pattern differs from this handle's last one, compared
 * entry by entry) and factorize. 0-based compressed columns (rows with
 * compressed_rows); indices within a column sorted, no duplicates;
 * nnz = col_ptr[n]. */
int32_t rs_update_i32(rs_solver *s, int64_t n, const int32_t *col_ptr,
                      const int32_t *row_idx, const double *values);
int32_t rs_update_i64(rs_solver *s, int64_t n, const int64_t *col_ptr,
                      const int64_t *row_idx, const double *values);

/* Solve A X = B in place: nrhs columns of length n, column j at b + j * ldb. */
int32_t rs_solve(rs_solver *s, int64_t nrhs, double *b, int64_t ldb);

/* Hager-Higham estimate of 1 / (||A||_1 ||A^-1||_1) with the current factor. */
int32_t rs_rcond_estimate(rs_solver *s, double *rcond);

int32_t rs_get_info(const rs_solver *s, rs_info *info);

/* Last error of the handle ("" after a success); valid until the next call. */
const char *rs_last_error(const rs_solver *s);

/* Static name of a status code. */
const char *rs_status_name(int32_t status);

/* Free the analyses kept for share_analysis. */
void rs_clear_shared_analyses(void);

#ifdef __cplusplus
}
#endif

#endif /* RUSTSOLVE_H */
