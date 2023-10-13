"""Inverses provided data using Minimum Fisher Regularisation (MFR) scheme.

To speed up the inversion process, the MFR scheme is implemented using :obj:`cupy` library
which allows to use GPU for matrix operations. If CUDA is not available, the code will fall back
to Numpy and Scipy compatible functions.
"""
import warnings
from datetime import timedelta
from time import time

import numpy as np
from scipy.sparse import issparse, spmatrix
from sksparse.cholmod import cholesky

from cherab.phix.inversion import Lcurve, _SVDBase

from ..tools import Spinner

try:
    # if CUDA is available:
    from cupy import arange, asarray, ndarray, ones_like, sqrt, zeros_like
    from cupy.linalg import inv
    from cupyx.scipy.sparse import csr_matrix, diags
    from cupyx.scipy.sparse.linalg import eigsh
except ImportError:
    warnings.warn(
        "Cupy is not available. Using Numpy and Scipy compatible functions instead.", stacklevel=1
    )
    from numpy import arange, asarray, ndarray, ones_like, sqrt, zeros_like
    from scipy.linalg import inv
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import eigsh

__all__ = ["Mfr"]


class Mfr:
    """Inverses provided data using Minimum Fisher Regularisation (MFR) scheme.

    The considered inverse problem is the tomographic reconstruction of the emissivity:

    .. math::

        T\\varepsilon = s,

    where :math:`T\\in\\mathbb{R}^{m\\times n}` is the geometry matrix,
    :math:`\\varepsilon\\in\\mathbb{R}^n` is the solution emissivity vector,
    and :math:`s\\in\\mathbb{R}^m` is the signal given data vecotr.

    The solution is usually calculated by the least square method, which is defined by

    .. math::

        \\varepsilon_\\text{ls} := \\text{argmin} \\{ ||T\\varepsilon - s||^2 \\}.

    Since this problem is often ill-posed, one requires a regularisation to be solved by adding
    an objective functional :math:`O(\\varepsilon)` to the least square problem:

    .. math::

        \\varepsilon_\\lambda
            := \\text{argmin} \\{ ||T\\varepsilon - s||^2 + \\lambda O(\\varepsilon) \\},

    where :math:`\\lambda` is a regularisation parameter.

    Parameters
    ----------
    gmat
        matrix :math:`T` of the forward problem (geometry matrix, ray transfer matrix, etc.)
    dmats
        list of pairs of derivative matrices :math:`D_i` and :math:`D_j` along to :math:`i` and
        :math:`j` coordinate directions, respectively
    data
        given data for inversion calculation, by default None
    """

    def __init__(self, gmat: np.ndarray, dmats: list[tuple[spmatrix, spmatrix]], data=None):
        # validate arguments
        gmat = asarray(gmat, dtype=float)
        if gmat.ndim != 2:
            raise ValueError("gmat must be a 2D array")

        if not isinstance(dmats, list):
            raise TypeError("dmats must be a list of tuples")
        for dmat1, dmat2 in dmats:
            if not issparse(dmat1):
                raise TypeError("one of the matrices in dmats is not a scipy sparse matrix")
            if not issparse(dmat2):
                raise TypeError("one of the matrices in dmats is not a scipy sparse matrix")

        # set matrix attributes
        self._gmat = gmat
        self._dmats = dmats

        # set data attribute
        self.data = data

    @property
    def gmats(self) -> ndarray:
        """Geometry matrix :math:`T` of the forward problem."""
        return self._gmat

    @property
    def dmats(self) -> list[tuple[spmatrix, spmatrix]]:
        """List of pairs of derivative matrices :math:`D_i` and :math:`D_j` along to :math:`i` and
        :math:`j` coordinate directions, respectively.
        """
        return self._dmats

    @property
    def data(self) -> np.ndarray:
        """Given data for inversion calculation."""
        return self._data

    @data.setter
    def data(self, value):
        data = np.asarray(value, dtype=float)
        if data.ndim != 1:
            raise ValueError("data must be a vector.")
        if data.size != self._gmat.shape[0]:
            raise ValueError("data size must be the same as the number of rows of U matrix")
        self._data = data

    def solve(
        self,
        x0: np.ndarray | None = None,
        derivative_weights: float | list[float] | None = None,
        eps: float = 1.0e-6,
        bounds: tuple[float, float] = (-20.0, 2.0),
        tol: float = 1e-3,
        miter: int = 20,
        regularizer: _SVDBase = Lcurve,
        **kwargs,
    ) -> tuple[ndarray, dict]:
        """Solves the inverse problem using MFR scheme.

        Parameters
        ----------
        x0
            initial solution vector, by default ones vector
        derivative_weights
            allows to specify anisotropy by assign weights for each matrix, by default ones vector
        eps
            small number to avoid division by zero, by default 1e-6
        bounds
            bounds of log10 of regularization parameter, by default (-20.0, 2.0).
        tol
            tolerance for solution convergence, by default 1e-3
        miter
            maximum number of MFR iterations, by default 20
        regularizer
            regularizer class to use, by default :obj:`cherab.phix.inversion.Lcurve`
        **kwargs
            additional keyword arguments passed to the regularizer class's :meth:`solve` method
        """
        # validate regularizer
        if not issubclass(regularizer, _SVDBase):
            raise TypeError("regularizer must be a subclass of _SVDBase")

        # check initial solution
        if x0 is None:
            x0 = np.ones(self._gmat.shape[1])
        elif isinstance(x0, np.ndarray):
            if x0.ndim != 1:
                raise ValueError("Initial solution must be a 1D array")
            if x0.shape[0] != self._gmat.shape[1]:
                raise ValueError("Initial solution must have same size as the rows of gmat")
        else:
            raise TypeError("Initial solution must be a numpy array")

        # MFR loop
        niter = 0
        status = {}
        start_time = time()
        while niter < miter or not self._converged:
            with Spinner(f"{niter}-th MFR iteration", timer=True) as sp:
                # compute regularisation matrix
                hmat = self.regularisation_matrix(
                    x0, eps=eps, derivative_weights=derivative_weights
                )

                # compute SVD components
                singular, u_vecs, basis = self._compute_svd(hmat)

                # find optimal solution using regularizer class
                reg = regularizer(singular, u_vecs, basis, data=self._data)
                x, _ = reg.solve(bounds=bounds, **kwargs)

                # check convergence
                error = np.linalg.norm(x - x0)
                self._converged = error < tol

                # update solution
                x0 = x

                # TODO: store regularizer object at each iteration

                sp.ok()

            niter += 1

        elapsed_time = time() - start_time
        status["elapsed_time"] = elapsed_time
        status["niter"] = niter

        print(f"Total elapsed time: {timedelta(seconds=elapsed_time)}")

        return x, status

    def regularisation_matrix(
        self,
        x: np.ndarray,
        eps: float = 1.0e-6,
        derivative_weights: list[float] | None = None,
    ):
        """Computes nonlinear regularisation matrix from provided derivative matrices and a solution
        vector.

        Multiple derivative matrices can be used allowing to combine matrices computed by
        different numerical schemes.

        Each matrix can have different weight coefficients assigned to introduce anisotropy.

        The expression of the regularisation matrix :math:`H(\\varepsilon)` is:

        .. math::

            H(\\varepsilon)
                = \\sum_{i,j} \\alpha_{ij} D_i^\\mathsf{T} W D_j

        where :math:`D_i` and :math:`D_j` are derivative matrices along to
        :math:`i` and :math:`j` coordinate directions, respectively, :math:`\\alpha_{ij}` is the
        anisotropic coefficient, and :math:`W` is the diagonal weight matrix defined as
        the inverse of :math:`\\varepsilon_i`:

        .. math::

            W_{ij} = \\frac{\\delta_{ij}}{\\max{\\varepsilon_i, \\epsilon_0}},

        where :math:`\\varepsilon_i` is the i-th element of the solution vector
        :math:`\\varepsilon`, and :math:`\\epsilon_0` is a small number to avoid division by zero
        and to push the solution to be positive.

        Parameters
        ----------
        x
            solution vector :math:`\\varepsilon`
        eps
            small number to avoid division by zero, by default 1e-6
        derivative_weights
            allows to specify anisotropy by assign weights for each matrix, by default ones vector

        Returns
        -------
        :obj:`scipy.sparse.csr_array` | :obj:`cupyx.scipy.sparse.csr_array`
            regularisation matrix :math:`H(\\varepsilon)`
        """
        # validate eps
        if eps <= 0:
            raise ValueError("eps must be positive small number")

        # set weighting matrix
        w = zeros_like(x)
        w[x > eps] = 1 / x[x > eps]
        w[x <= eps] = 1 / eps
        w = diags(w)

        if derivative_weights is None:
            derivative_weights = [1.0] * len(self._dmats)
        elif len(derivative_weights) != len(self._dmats):
            raise ValueError(
                "Number of derivative weight coefficients must be equal to number of derivative matrices"
            )

        regularisation = csr_matrix(self._dmats[0][0].shape, dtype=float)

        for (dmat1, dmat2), aniso in zip(self._dmats, derivative_weights, strict=False):
            regularisation += aniso * csr_matrix(dmat1).T @ w @ csr_matrix(dmat2)

        return regularisation

    def _compute_svd(self, hmat: csr_matrix):
        """Computes singular value decomposition of the regularisation matrix."""
        # cholesky decomposition of H
        factor = cholesky(hmat)
        L_mat = factor.L()

        # compute the fill-reducing permutation matrix P
        P_vec = factor.P()
        rows = arange(len(P_vec))
        data = ones_like(rows)
        P_mat = csr_matrix((data, (rows, P_vec)), dtype=np.int8)

        # compute A = T P^T L^{-T} and AA^T
        Lt_inv = inv(asarray(L_mat.toarray())).T
        A_mat = csr_matrix(self._gmat) @ P_mat.T @ Lt_inv  # A = T P^T L^{-T}
        At_mat = A_mat.T
        AAt = A_mat @ At_mat

        # compute eigenvalues and eigenvectors of AA^T
        eigvals, eigvecs = eigsh(AAt, k=AAt.shape[0] - 1, which="LM", return_eigenvectors=True)

        # sort eigenvalues and eigenvectors in descending order
        decend_index = eigvals.argsort()[::-1]
        eigvals = eigvals[decend_index]
        eigvecs = eigvecs[:, decend_index]

        # calculate singular values and left vectors (w/o zero eigenvalues)
        singular = sqrt(eigvals[eigvals > 0])
        u_vecs = eigvecs[:, eigvals > 0]

        # compute right singular vectors
        v_mat = At_mat @ u_vecs @ diags(1 / singular)

        # compute inverted solution basis
        basis = P_mat.T @ Lt_inv @ v_mat

        return singular, u_vecs, basis
