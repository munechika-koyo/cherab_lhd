from numpy cimport ndarray, int32_t, uint8_t
from raysect.core cimport BoundingBox3D, Point3D, AffineMatrix3D
from raysect.core.math.spatial.kdtree3d cimport KDTree3DCore


cdef class TetraMeshData(KDTree3DCore):

    cdef:
        ndarray _vertices
        ndarray _tetrahedra
        double[:, ::1] vertices_mv
        int32_t[:, ::1] tetrahedra_mv
        int32_t tetrahedra_id
        int32_t i1, i2, i3, i4
        double alpha, beta, gamma, delta
        bint _cache_available
        double _cached_x
        double _cached_y
        double _cached_z
        bint _cached_result

    cpdef Point3D vertex(self, int index)

    cpdef ndarray tetrahedron(self, int index)

    cpdef Point3D barycenter(self, int index)

    cpdef double volume(self, int index)

    cpdef double volume_total(self)

    cdef double _volume(self, int index)

    cdef object _filter_tetrahedra(self)

    cdef BoundingBox3D _generate_bounding_box(self, int32_t tetrahedra)

    cpdef BoundingBox3D bounding_box(self, AffineMatrix3D to_world)

    cdef uint8_t _read_uint8(self, object file)

    cdef bint _read_bool(self, object file)
