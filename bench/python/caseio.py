"""Read and write exported linear systems.

A case is a directory with
  meta.json            n, nnz, nrhs, description and provenance
  indptr.i64 / indices.i64 / data.f64   full (both triangles) CSR of A
  rhs.f64              n x nrhs, column-major
  xref.f64             the reference backend's solutions, n x nrhs (optional)
All binaries are little-endian, no header.
"""
import json
import os

import numpy as np


def save_case(path, indptr, indices, data, rhs, xref=None, meta=None):
    os.makedirs(path, exist_ok=True)
    indptr = np.ascontiguousarray(indptr, dtype="<i8")
    indices = np.ascontiguousarray(indices, dtype="<i8")
    data = np.ascontiguousarray(data, dtype="<f8")
    rhs = np.asarray(rhs, dtype="<f8")
    if rhs.ndim == 1:
        rhs = rhs[:, None]
    n = indptr.size - 1
    assert rhs.shape[0] == n, (rhs.shape, n)
    indptr.tofile(os.path.join(path, "indptr.i64"))
    indices.tofile(os.path.join(path, "indices.i64"))
    data.tofile(os.path.join(path, "data.f64"))
    np.asfortranarray(rhs).T.ravel().tofile(os.path.join(path, "rhs.f64"))
    m = dict(meta or {})
    m.update(n=int(n), nnz=int(indices.size), nrhs=int(rhs.shape[1]),
             has_xref=xref is not None)
    if xref is not None:
        xref = np.asarray(xref, dtype="<f8")
        if xref.ndim == 1:
            xref = xref[:, None]
        np.asfortranarray(xref).T.ravel().tofile(os.path.join(path, "xref.f64"))
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(m, f, indent=1)


def load_case(path):
    import scipy.sparse as sp
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    n, m = meta["n"], meta["nrhs"]
    indptr = np.fromfile(os.path.join(path, "indptr.i64"), dtype="<i8")
    indices = np.fromfile(os.path.join(path, "indices.i64"), dtype="<i8")
    data = np.fromfile(os.path.join(path, "data.f64"), dtype="<f8")
    A = sp.csr_matrix((data, indices, indptr), shape=(n, n))
    rhs = np.fromfile(os.path.join(path, "rhs.f64"), dtype="<f8").reshape(m, n).T
    xref = None
    if meta.get("has_xref"):
        xref = np.fromfile(os.path.join(path, "xref.f64"), dtype="<f8").reshape(m, n).T
    return A, rhs, xref, meta
