"""Sparse direct solves (Cholesky, LDL^T, LU) with a Rust core (faer).

>>> import scipy.sparse as sp, numpy as np, rustsolve
>>> A = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(5, 5), format="csr")
>>> f = rustsolve.factorize(A)            # Cholesky
>>> np.allclose(A @ f.solve(np.ones(5)), 1)
True

Nothing is perturbed to make a factorization succeed. A failure raises a
:class:`SolverError` subclass, and the factor is gone afterwards: ``solve``
raises :class:`NotFactorizedError` instead of returning a solution of an
earlier matrix.
"""

from __future__ import annotations

import operator

import numpy as np

from ._rustsolve import Factor as _Factor
from ._rustsolve import SolverError, __version__, clear_shared_analyses, has_nested_dissection

__all__ = [
    "Factor", "factorize", "solve", "SolverError", "clear_shared_analyses",
    "InvalidInputError", "NotSymmetricError", "NonFiniteInputError", "PatternChangedError",
    "NotFactorizedError", "NotPositiveDefiniteError", "SingularMatrixError", "__version__",
    "has_nested_dissection",
]


class InvalidInputError(SolverError, ValueError):
    """Malformed pattern or arguments."""


class NotSymmetricError(SolverError, ValueError):
    """A symmetric factorization got a matrix whose triangles differ."""


class NonFiniteInputError(SolverError, ValueError):
    """NaN or infinity in the matrix."""


class PatternChangedError(SolverError):
    """The pattern differs from the analyzed one (use ``update``)."""


class NotFactorizedError(SolverError):
    """No valid factorization (none yet, or the last one failed)."""


class NotPositiveDefiniteError(SolverError, np.linalg.LinAlgError):
    """Cholesky met a pivot <= 0; ``step`` and ``column`` say where."""


class SingularMatrixError(SolverError, np.linalg.LinAlgError):
    """Zero pivot, no pivot, or numerically singular."""


_BY_NAME = {
    "INVALID_INPUT": InvalidInputError,
    "STRUCTURALLY_SINGULAR": SingularMatrixError,
    "NOT_SYMMETRIC": NotSymmetricError,
    "NON_FINITE_INPUT": NonFiniteInputError,
    "PATTERN_CHANGED": PatternChangedError,
    "NOT_ANALYZED": NotFactorizedError,
    "NOT_FACTORIZED": NotFactorizedError,
    "NOT_POSITIVE_DEFINITE": NotPositiveDefiniteError,
    "ZERO_PIVOT": SingularMatrixError,
    "NUMERICALLY_SINGULAR": SingularMatrixError,
    "SINGULAR": SingularMatrixError,
}


def _raise(e: SolverError):
    name, message, step, column = (tuple(e.args) + (None,) * 4)[:4]
    cls = _BY_NAME.get(name, SolverError)
    err = cls(message)
    err.code, err.step, err.column = name, step, column
    raise err from None


def _real(x, what):
    # converting complex values to float64 would only warn and drop the
    # imaginary part, giving the solution of another problem; object arrays
    # can hold NumPy complex scalars that np.iscomplexobj does not see
    dtype = getattr(x, "dtype", None)
    if dtype is None:
        dtype = np.asarray(x).dtype
    if dtype.kind not in "biuf":
        raise TypeError(f"{what} must hold real numbers, not dtype {dtype}; "
                        "rustsolve solves real systems only")


def _index_array(x, what):
    # casting would turn 0.9 into 0, or drop an imaginary part: another pattern
    x = np.asarray(x)
    if x.dtype.kind not in "iu" and not (x.size == 0 and x.dtype.kind in "biuf"):
        raise TypeError(f"{what} must hold integers, not dtype {x.dtype}")
    return x


def _arrays(A):
    """(n, indptr, indices, data, compressed_rows) of a scipy.sparse matrix or
    of a tuple (indptr, indices, data, n, "csr" | "csc")."""
    if isinstance(A, tuple):
        indptr, indices, data, n, fmt = A
        if fmt not in ("csr", "csc"):
            raise ValueError(f"the format of a tuple must be 'csr' or 'csc', not {fmt!r}")
        try:
            n = operator.index(n)  # int(2.7) would be 2
        except TypeError:
            raise TypeError(f"n must be an integer, not {type(n).__name__}") from None
        if n < 0:
            raise ValueError(f"n must not be negative, not {n}")
        data = np.asarray(data)
        _real(data, "A")
        return n, _index_array(indptr, "indptr"), _index_array(indices, "indices"), data, fmt == "csr"
    _real(A, "A")  # scipy.sparse matrices carry a dtype
    fmt = getattr(A, "format", None)
    if fmt not in ("csr", "csc"):
        if not hasattr(A, "tocsr"):
            raise TypeError("A must be a scipy.sparse matrix or (indptr, indices, data, n, 'csr'|'csc')")
        A, fmt = A.tocsr(), "csr"
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, not {A.shape}")
    if not A.has_canonical_format:
        A = A.copy()
        A.sum_duplicates()  # also sorts the indices
    return A.shape[0], A.indptr, A.indices, A.data, fmt == "csr"


class Factor:
    """Factorization of one matrix at a time.

    Parameters
    ----------
    kind : {"cholesky", "ldlt", "lu"}
    ordering : {"amd", "nd"}
        Fill-reducing ordering of the symmetric kinds ("nd" if built with it).
    threads, solve_threads : int
        1 = sequential (default), 0 = all cores.
    check_symmetry : bool
        Symmetric kinds: refuse an entry that differs from its mirror (a
        mirror that is not stored counts as zero). Without it, the
        factorization reads the lower triangle (CSC) or upper triangle (CSR).
    share_analysis : bool
        Reuse the analysis of an identical pattern made by another Factor.
    """

    def __init__(self, kind="cholesky", *, ordering="amd", nodal="auto", threads=1, solve_threads=1,
                 check_symmetry=True, symmetry_tol=1e-12, singular_pivot_ratio=None, min_rcond=None,
                 share_analysis=False):
        self._opts = dict(kind=kind, ordering=ordering, nodal=nodal, threads=threads,
                          solve_threads=solve_threads, check_symmetry=check_symmetry,
                          symmetry_tol=symmetry_tol, singular_pivot_ratio=singular_pivot_ratio,
                          min_rcond=min_rcond, share_analysis=share_analysis)
        # validates the options now; rebuilt for CSR input in update
        try:
            self._f = _Factor(compressed_rows=False, **self._opts)
        except SolverError as e:
            _raise(e)
        self._rows = False
        self._factorized = False

    def update(self, A):
        """Factorize A; the analysis is redone only when A's pattern is new."""
        # until this update succeeds there is no factor to solve with
        self._factorized = False
        n, indptr, indices, data, rows = _arrays(A)
        if rows != self._rows:
            try:
                self._f = _Factor(compressed_rows=rows, **self._opts)
            except SolverError as e:
                _raise(e)
            self._rows = rows
        try:
            self._f.update(n, np.ascontiguousarray(indptr, dtype=np.int64),
                           np.ascontiguousarray(indices, dtype=np.int64),
                           np.ascontiguousarray(data, dtype=np.float64))
        except SolverError as e:
            _raise(e)
        self._factorized = True
        return self

    def solve(self, b):
        """x with A x = b, for b of shape (n,) or (n, k)."""
        if not self._factorized:
            raise NotFactorizedError("no matrix factorized yet")
        b = np.asarray(b)
        _real(b, "b")
        try:
            return self._f.solve(np.asarray(b, dtype=np.float64))
        except SolverError as e:
            _raise(e)

    __call__ = solve

    def rcond(self) -> float:
        """Reciprocal 1-norm condition estimate (a few solves)."""
        if not self._factorized:
            raise NotFactorizedError("no matrix factorized yet")
        try:
            return self._f.rcond()
        except SolverError as e:
            _raise(e)

    @property
    def info(self) -> dict:
        return dict(self._f.info)


def factorize(A, kind="cholesky", **options) -> Factor:
    """Factorize a sparse matrix; see :class:`Factor` for the options."""
    return Factor(kind, **options).update(A)


def solve(A, b, kind="cholesky", **options):
    """Solve A x = b once."""
    return factorize(A, kind, **options).solve(b)
