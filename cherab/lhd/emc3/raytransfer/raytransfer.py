"""
This module offers the helper function to easily set raytransfer material
"""
from __future__ import annotations

from cherab.lhd.emc3.geometry import TomographyZone
from cherab.lhd.emc3.raytransfer import Discrete3DMeshRayTransferEmitter
from raysect.core.math import translate
from raysect.optical import World
from raysect.primitive import Cylinder

__all__ = ["load_rte_emc3"]


# Constants
RMIN, RMAX = 2.0, 5.5  # [m]
ZMIN, ZMAX = -1.6, 1.6


def load_rte_emc3(parent: World, bins: int | None = None) -> Cylinder:
    """
    Helper function of loding RayTransfer Emitter using :obj:`.Discrete3DMeshRayTransferEmitter`

    Parameters
    ----------
    parent
        raysect world Node
    bins
        the number of grids, by default ``14 * 18``

    Returns
    -------
    :obj:`~raysect.primitive.cylinder.Cylinder`
        primitive of cylinder
    """

    bins = bins or 14 * 18  # 14 zones x 18 degrees

    # Load index function
    index_func = TomographyZone()

    # material as emitter
    material = Discrete3DMeshRayTransferEmitter(index_func, bins, integration_step=0.001)

    # primitive using cylinder
    shift = translate(0, 0, ZMIN)
    emitter = Cylinder(RMAX, ZMAX - ZMIN, transform=shift, parent=parent, material=material)

    return emitter


if __name__ == "__main__":
    pass
