"""Module to offer Discrete3D interpolation for EMC3-EIRENE in LHD."""
cimport cython
from libc.limits cimport INT_MIN
from libc.math cimport M_PI, atan2, cos, hypot, sin
from numpy cimport uint32_t
from raysect.core.math.point cimport new_point3d

__all__ = ["Discrete3DMesh"]


# converted some degree to radian
cdef double PHI_18, PHI_36, PHI_54, PHI_72
PHI_18 = 18.0 * M_PI / 180.0
PHI_36 = 36.0 * M_PI / 180.0
PHI_54 = 54.0 * M_PI / 180.0
PHI_72 = 72.0 * M_PI / 180.0


cdef class Discrete3DMesh(IntegerFunction3D):
    """Discrete interpolator for indices on a 3-D EMC3-EIRENE cell.

    This class offers the callable taking :math:`(X, Y ,Z)` positional arguments and returning
    the corresponding cell index.

    One cell consists of 8 vertices forming a cubic-like shape (not appropriate cubic),
    and each cell is divided six tetrahedra to create a tetrahedral mesh.

    EMC3-EIRENE cells in LHD have specific periodicity along to the toroidal direction.
    The region between [0, 72] degree has periodicity in LHD's equilibrium configuration.
    Moreover, there are 4 regions: [0, 18], [18, 36], [36, 54], [54, 72] degree in toroidal.

    Toroidal angle :math:`\\phi` is converted like :math:`\\phi_r \\equiv \\phi \\% 72^\\circ`.
    The returned cell index depends on which region includes the point.
    The relationship between cell indices and :math:`(R, Z, \\phi_r)` in each toroidal regions
    is represented as follows:

    * :math:`\\phi_r \\in [0, 18]`:  `indices1` at :math:`(R, Z, \\phi_r)`
    * :math:`\\phi_r \\in (18, 36]`: `indices2` at :math:`(R, -Z, 36^\\circ - \\phi_r)`
    * :math:`\\phi_r \\in (36, 54]`: `indices3` at :math:`(R, Z, \\phi_r - 36^\\circ)`
    * :math:`\\phi_r \\in (54, 72]`: `indices4` at :math:`(R, -Z, 72^\\circ - \\phi_r)`

    `indices1` & `indices2` must be specified as 1-D numpy array, the others are optional.
    If they are None, `indices3` & `indices4` are referred to `indices1` & `indices2`,
    respectively.

    If the specified point is outside the defined tetrahedral mesh, this callble always returns -1.

    To optimise the lookup of tetrahedra, acceleration structure (a KD-Tree) is used from the
    specified instance of `.TetraMeshData`.

    Parameters
    ----------
    tetra : `.TetraMeshData`
        `.TetraMeshData` instances.
    indices1 : ndarray[uint32, ndim=1]
        1-D EMC3-EIRENE's cell indices array which is used in [0, 18] degree in toroidal.
    indices2 : ndarray[uint32, ndim=1]
        1-D EMC3-EIRENE's cell indices array which is used in (18, 36] degree in toroidal.
    indices3 : ndarray[uint32, ndim=1], optional
        1-D EMC3-EIRENE's cell indices array which is used in (36, 54] degree in toroidal,
        if None, this is referred to `indices1`
    indices4 : ndarray[uint32, ndim=1], optional
        1-D EMC3-EIRENE's cell indices array which is used in (54, 72] degree in toroidal,
        if None, this is referred to `indices2`
    """
    def __init__(
        self,
        TetraMeshData tetra,
        uint32_t[::1] indices1,
        uint32_t[::1] indices2,
        uint32_t[::1] indices3 = None,
        uint32_t[::1] indices4 = None,
    ):
        # populate internal attributes
        self._tetra_mesh = tetra
        self._indices1_mv = indices1
        self._indices2_mv = indices2

        if indices3 is not None:
            self._indices3_mv = indices3
        else:
            self._indices3_mv = indices1

        if indices4 is not None:
            self._indices4_mv = indices4
        else:
            self._indices4_mv = indices2

    def __getstate__(self):
        return (
            self._tetra_mesh,
            self._indices1_mv,
            self._indices2_mv,
            self._indices3_mv,
            self._indices4_mv,
        )

    def __setstate__(self, state):
        (
            self._tetra_mesh,
            self._indices1_mv,
            self._indices2_mv,
            self._indices3_mv,
            self._indices4_mv,
        ) = state

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    @property
    def tetra_mesh(self):
        """`.TetraMeshData`: Tetrahedral mesh instance
        """
        return self._tetra_mesh

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef int evaluate(self, double x, double y, double z) except? INT_MIN:

        cdef:
            double _x, _y, _z, _r, _phi

        # identify in which region the point exists
        _r = hypot(x, y)
        _phi = atan2(y, x)

        # _phi must be in [0, PHI_72]
        if _phi < 0:
            _phi = (_phi + 2.0 * M_PI) % PHI_72
        else:
            _phi = _phi % PHI_72

        if _phi <= PHI_18:
            _x = _r * cos(_phi)
            _y = _r * sin(_phi)
            _z = z

            if self._tetra_mesh.is_contained(new_point3d(_x, _y, _z)):
                return self._indices1_mv[<int>(self._tetra_mesh.tetrahedra_id // 6)]

        elif PHI_18 < _phi <= PHI_36:
            _phi = PHI_36 - _phi
            _x = _r * cos(_phi)
            _y = _r * sin(_phi)
            _z = z * -1.0

            if self._tetra_mesh.is_contained(new_point3d(_x, _y, _z)):
                return self._indices2_mv[<int>(self._tetra_mesh.tetrahedra_id // 6)]

        elif PHI_36 < _phi <= PHI_54:
            _phi = _phi - PHI_36
            _x = _r * cos(_phi)
            _y = _r * sin(_phi)
            _z = z

            if self._tetra_mesh.is_contained(new_point3d(_x, _y, _z)):
                return self._indices3_mv[<int>(self._tetra_mesh.tetrahedra_id // 6)]

        else:
            _phi = PHI_72 - _phi
            _x = _r * cos(_phi)
            _y = _r * sin(_phi)
            _z = z * -1.0

            if self._tetra_mesh.is_contained(new_point3d(_x, _y, _z)):
                return self._indices4_mv[<int>(self._tetra_mesh.tetrahedra_id // 6)]

        # If the point is outside the mesh
        return -1
