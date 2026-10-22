"""Python layer: scipy.sparse in, numpy out, errors as exception classes."""

import numpy as np
import pytest

sp = pytest.importorskip("scipy.sparse")
spl = pytest.importorskip("scipy.sparse.linalg")

import rustsolve


def laplace2d(nx, ny, shift=1e-2, sigma=None):
    """5-point Laplacian with a diagonal shift (SPD), optional cell weights."""
    ex = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(nx, nx))
    ey = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(ny, ny))
    A = sp.kron(sp.identity(ny), ex) + sp.kron(ey, sp.identity(nx)) + shift * sp.identity(nx * ny)
    if sigma is not None:
        d = sp.diags(np.sqrt(sigma))
        A = d @ A @ d
    return A.tocsr()


@pytest.mark.parametrize("kind", ["cholesky", "ldlt", "lu"])
@pytest.mark.parametrize("fmt", ["csr", "csc", "coo"])
def test_matches_scipy(kind, fmt):
    rng = np.random.default_rng(0)
    A = laplace2d(30, 20, sigma=10.0 ** rng.uniform(-3, 3, 600)).asformat(fmt)
    B = rng.standard_normal((600, 4))
    X = rustsolve.factorize(A, kind).solve(B)
    Xs = spl.spsolve(A.tocsc(), B)
    assert X.shape == B.shape
    assert np.max(np.abs(X - Xs)) / np.max(np.abs(Xs)) < 1e-8
    x1 = rustsolve.solve(A, B[:, 0], kind)
    assert x1.shape == (600,)
    assert np.allclose(x1, X[:, 0], rtol=1e-12, atol=0)


def test_update_reuses_the_analysis_and_uses_new_values():
    A = laplace2d(25, 25)
    A2 = laplace2d(25, 25, shift=5.0)
    b = np.ones(A.shape[0])
    f = rustsolve.factorize(A)
    assert f.info["reanalyzed"]
    f.update(A2)
    assert not f.info["reanalyzed"] and f.info["factorizations"] == 2
    assert np.allclose(A2 @ f.solve(b), b)


def test_lu_on_nonsymmetric_csr_and_csc():
    A = laplace2d(15, 15) + 0.3 * sp.diags([1.0], [1], shape=(225, 225))
    b = np.arange(225.0)
    xs = spl.spsolve(A.tocsc(), b)
    for M in (A.tocsr(), A.tocsc()):
        x = rustsolve.solve(M, b, "lu")
        assert np.allclose(x, xs, rtol=1e-10, atol=1e-12)
    with pytest.raises(rustsolve.NotSymmetricError):
        rustsolve.factorize(A)


def test_errors_and_no_stale_factor():
    A = laplace2d(10, 10)
    f = rustsolve.factorize(A)
    with pytest.raises(rustsolve.NotPositiveDefiniteError) as e:
        f.update(-A)
    assert e.value.code == "NOT_POSITIVE_DEFINITE" and e.value.column is not None
    with pytest.raises(rustsolve.NotFactorizedError):
        f.solve(np.ones(100))
    bad = A.copy()
    bad.data[3] = np.nan
    with pytest.raises(rustsolve.NonFiniteInputError):
        f.update(bad)
    # an input refused before it reaches the solver drops the factor too
    f.update(A)
    with pytest.raises(ValueError):
        f.update(sp.csr_matrix(np.ones((3, 4))))
    with pytest.raises(rustsolve.NotFactorizedError):
        f.solve(np.ones(100))
    assert isinstance(rustsolve.NotPositiveDefiniteError("x"), np.linalg.LinAlgError)
    with pytest.raises(ValueError):
        rustsolve.Factor("qr")


def test_complex_input_is_refused():
    # converting to float64 would only warn and drop the imaginary part
    A = laplace2d(5, 5)
    with pytest.raises(TypeError):
        rustsolve.factorize(A.astype(complex))
    f = rustsolve.factorize(A)
    with pytest.raises(TypeError):
        f.solve(np.ones(25) + 1j)
    with pytest.raises(TypeError):
        rustsolve.Factor().update((A.indptr, A.indices, A.data + 0j, 25, "csr"))
    # NumPy complex scalars in object arrays, which np.iscomplexobj misses
    obj = np.array([np.complex128(v) for v in A.data], dtype=object)
    with pytest.raises(TypeError):
        rustsolve.Factor().update((A.indptr, A.indices, obj, 25, "csr"))
    with pytest.raises(TypeError):
        f.solve(np.array([np.complex128(1)] * 25, dtype=object))


def test_tuple_format_is_checked():
    # an unknown format is refused rather than read as CSC (the transpose of CSR)
    B = sp.csr_matrix(np.array([[2.0, 1.0], [0.0, 3.0]]))
    t = lambda fmt: (B.indptr, B.indices, B.data, 2, fmt)
    assert np.allclose(rustsolve.Factor("lu").update(t("csr")).solve(np.array([3.0, 3.0])), [1, 1])
    for fmt in ("CSR", "oops"):
        with pytest.raises(ValueError):
            rustsolve.Factor("lu").update(t(fmt))
    # fractional or complex indices and a fractional n are refused, not cast to
    # another pattern; a refused update leaves nothing to solve with
    f = rustsolve.Factor("lu").update(t("csr"))
    bad = [(B.indptr, B.indices + 0.9, B.data, 2, "csr"), (B.indptr, B.indices + 5j, B.data, 2, "csr"),
           (B.indptr.astype(float), B.indices, B.data, 2, "csr"), (B.indptr, B.indices, B.data, 2.7, "csr")]
    for tup in bad:
        with pytest.raises(TypeError):
            f.update(tup)
        with pytest.raises(rustsolve.NotFactorizedError):
            f.solve(np.array([3.0, 3.0]))
    with pytest.raises(ValueError):
        f.update((B.indptr, B.indices, B.data, -2, "csr"))


def neumann2d(nx, ny):
    """Graph Laplacian of a grid: rows sum to zero, constant null space."""
    def lap1(m):
        L = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(m, m)).tolil()
        L[0, 0] = L[m - 1, m - 1] = 1.0
        return L.tocsr()
    return (sp.kron(sp.identity(ny), lap1(nx)) + sp.kron(lap1(ny), sp.identity(nx))).tocsr()


def test_singular_neumann_matrix_is_refused():
    A = neumann2d(12, 12)
    assert np.allclose(A @ np.ones(144), 0)
    for kind in ("cholesky", "ldlt", "lu"):
        with pytest.raises((rustsolve.NotPositiveDefiniteError, rustsolve.SingularMatrixError)):
            rustsolve.factorize(A, kind)

