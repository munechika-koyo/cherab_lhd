"""Inverses provided data using Minimum Fisher Regularisation (MFR) scheme."""
from scipy.sparse import csr_array, dia_array
from tomotok.core.inversions.mfr import Mfr


class MFR(Mfr):
    """Inverses provided data using Minimum Fisher Regularisation (MFR) scheme.

    This class is a wrapper around :obj:`~tomotok.core.inversions.mfr.Mfr` class.
    Please refer to the documentation of the base class for more details.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def regularisation_matrix(
        self,
        derivatives: list[csr_array],
        weights: dia_array,
        weight_coefficients: list[float] | None = None,
        **kwargs,
    ):
        """Computes nonlinear regularisation matrix from provided derivative matrices and node
        weight factors determined by emissivity from previous iteration of the inversion loop.

        Multiple derivative matrices can be used allowing to combine matrices computed by
        different numerical schemes.

        Each matrix can have different weight coefficients assigned to introduce anisotropy.

        The expression of the regularisation matrix :math:`\\mathbf{H}` is:

        .. math::

            \\mathbf{H} = \\sum_{i,j} w_{ij} \\mathbf{D}_i^T \\mathbf{W} \\mathbf{D}_j,

        where :math:`\\mathbf{D}_i` and :math:`\\mathbf{D}_j` are derivative matrices along to
        :math:`i` and :math:`j` coordinate directions, respectively, :math:`w_{ij}` is the weight
        coefficient, and :math:`\\mathbf{W}` is the diagonal matrix of node weight factors which is
        determined by emissivity :math:`\\varepsilon_i` like:

        .. math::

            \\mathbf{W}_{ij} =
            \\begin{cases}
                \\frac{1}{\\varepsilon_i} \\cdot \\delta_{ij} & \\text{if } \\varepsilon_i > 0 \\\\
                W_\\text{max} \\cdot \\delta_{ij} & \\text{otherwise}
            \\end{cases}

        where :math:`W_\\text{max}` is the maximum value of :math:`\\mathbf{W}` or an user-defined
        value. This wight factor is updated after each iteration of the inversion loop.


        Parameters
        ----------
        derivatives
            a list of derivative matrices, with shape (# nodes, # nodes)
        w
            diagonal matrix with weight factors (#nodes, #nodes)
        weight_coefficients
            allows to specify anisotropy by assign weights for each matrix

        Returns
        -------
        :obj:`~scipy.sparse.csr_array`
            regularisation matrix
        """
        if weight_coefficients is None:
            weight_coefficients = [1.0] * len(derivatives)
        elif len(weight_coefficients) != len(derivatives):
            raise ValueError(
                "Number of weight coefficients must be equal to number of derivative matrices"
            )

        regularisation = csr_array(derivatives[0].shape, dtype=float)

        for (dmat1, dmat2), aniso in zip(derivatives, weight_coefficients, strict=False):
            regularisation += aniso * dmat1.T * weights * dmat2

        return regularisation
