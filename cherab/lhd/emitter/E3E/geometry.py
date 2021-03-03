import os
import numpy as np
import pickle
from matplotlib import pyplot as plt
from cherab.lhd.emitter.E3E.cython import Discrete3DMesh
from cherab.lhd.emitter.E3E.utils import read_E3E_grid, generate_faces

BASE = os.path.dirname(__file__)
GRID_PATH = os.path.join(BASE, "data", "grid-360.txt")
CELLGEO_PATH = os.path.join(BASE, "data", "CELL_GEO")


class EMC3:
    """EMC3-EIRINE class
    This class is implemented for EMC3's grids, physical cell indices, and tetrahedralization
    to enable raysect to handle them as an emitter.

    Parameters
    ----------
    zones : list, optional
        EMC3's zones labels, by default :math:`['zone0', 'zone1',..., 'zone4',
                                                'zone11', 'zone12', ..., 'zone15']`
        Basically, EMC3's grids corresponding to zone labels are used for the tetrahedralization.
    """

    def __init__(self, zones=None) -> None:
        # initialize properties
        self._zones = zones or [f"zone{i}" for i in [j for j in range(0, 4 + 1)] + [k for k in range(11, 15 + 1)]]
        self._r = None
        self._z = None
        self._num_radial = None
        self._num_poloidal = None
        self._num_toroidal = None
        self._num_cells = None
        self._vertices = None
        self._tetrahedra = None
        self._phys_cells = None

    def load_grids(self, path=None) -> None:
        """Load EMC3-Eirine grids (r, z) coordinates, which is classified by zones.
        The (r, z) coords. grouped by each zones are loaded as the type of dictonary.
        Labels of zones are in ['zone0',..., 'zone21'].
        The number of radial, poloindal toroidal and cells are also loaded.

        Parameters
        ----------
        path : str, optional
            path to the grids text file, by default ".../data/grid-360.txt"
        """

        # path to E3E grid file
        self.grid_path = path or GRID_PATH

        # load geometry cell's grids and numbers
        self._r, self._z, self._num_radial, self._num_poloidal, self._num_toroidal, self._num_cells = read_E3E_grid(self.grid_path)

    def load_cell_geo(self, path=None) -> None:
        """Load cell geometry indices
        EMC3 has many geometry cells, the number of which is defined :math:`num_cells` in each zones.
        However, EMC3 calculates the physical values such as plasma density in each cells combined several geometry cells,
        which is called "physical" cells. Their relationship between geomtry and physical cells indices is written in CELL_GEO file.

        Parameters
        ----------
        path : str, optional
            path to the CELL_GEO file, by default ".../data/CELL_GEO"
            The default file "CELL_GEO" has geometric indices in all zones.
        """
        # path to CELL_GEO file
        self.cellgeo_path = path or CELLGEO_PATH

        # load cell_geo indices
        with open(self.cellgeo_path, "r") as f:
            line = f.readline()
            num_geo, num_plasma, num_total = list(map(int, line.split()))
            line = f.readline()
            cell_geo = [int(x) - 1 for x in line.split()]  # for c language index format

        # classify all cell_geo indices by zone labels
        if self._num_cells is None:
            self.load_grids()

        self._phys_cells = {}
        start = 0
        for zone in self._num_cells.keys():
            num = self._num_cells[zone]
            self._phys_cells[zone] = cell_geo[start:start + num]
            start += num

    def tetrahedralization(self, zones=None):
        """tetrahedrarization
        This method has the function of generating tetrahedral vertices and indices valiables.
        The user can select zones storing grids coords. and tetrahedrarizate using them.

        Parameters
        ----------
        zones : list, optional
            containig zone labels, by default self.zones :math:`['zone0', 'zone1', ...]`
        """
        # set zones which are tetrahedrarizate
        zones = zones or self.zones
        if not isinstance(zones, list):
            raise TypeError("zones must be a list.")

        # load E3E vertices if they are not imported yet
        if self.r or self.z is None:
            self.load_grids()

        # check if all elements of zones arg is included in zones dict.keys.
        if not all([zone in self.num_cells.keys() for zone in zones]):
            raise ValueError("zones labels are contained in the property 'zones'.")

        # initialize local variables
        vertices = {}
        tetrahedra = {}

        for zone in zones:

            # set local variables: vertices and tetrahedral indices in each zones
            num_total = self._num_radial[zone] * self._num_poloidal[zone]
            vertices[zone] = np.zeros((num_total * self._num_toroidal[zone], 3), dtype=np.float64)
            tetrahedra[zone] = np.zeros((6 * self._num_cells[zone], 4), dtype=np.uint32)

            # gengerate vertices
            offset = 0
            for toroidal_angle in self._r[zone].keys():
                tor_angle_rad = toroidal_angle * np.pi / 180.0
                for row in range(num_total):
                    vertices[zone][row + offset, :] = (self._r[zone][toroidal_angle][row] * np.cos(tor_angle_rad), self._r[zone][toroidal_angle][row] * np.sin(tor_angle_rad), self._z[zone][toroidal_angle][row])
                offset += num_total

            # geomtric cell indices at one toroidal plane
            faces = generate_faces(self.num_radial, self.num_poloidal, zone=zone)

            # Divide a cell into six tetrahedra.
            tet_id = 0
            offset = 0
            for i in range(self._num_toroidal[zone] - 1):
                for face in faces:
                    tetrahedra[zone][tet_id, :] = np.array([face[0] + num_total, face[0], face[1], face[2]], dtype=np.uint32) + offset
                    tetrahedra[zone][tet_id + 1, :] = np.array([face[3] + num_total, face[3], face[0], face[2]], dtype=np.uint32) + offset
                    tetrahedra[zone][tet_id + 2, :] = np.array([face[1], face[1] + num_total, face[2] + num_total, face[0] + num_total], dtype=np.uint32) + offset
                    tetrahedra[zone][tet_id + 3, :] = np.array([face[2], face[2] + num_total, face[3] + num_total, face[0] + num_total], dtype=np.uint32) + offset
                    tetrahedra[zone][tet_id + 4, :] = np.array([face[0], face[2], face[3] + num_total, face[0] + num_total], dtype=np.uint32) + offset
                    tetrahedra[zone][tet_id + 5, :] = np.array([face[1], face[2], face[0] + num_total, face[2] + num_total], dtype=np.uint32) + offset

                    tet_id += 6
                offset += num_total

        # store as properties
        self._vertices = vertices
        self._tetrahedra = tetrahedra

    def generate_index_function(self, zones=None, save=True, path=None):

        # path to save function as pickel
        path = path or os.path.join(BASE, "data", "emc3_grid.pickel")

        # load CELL_GEO file if yet
        if self._phys_cells is None:
            self.load_cell_geo()

        # choose zones
        zones = zones or list(self.vertices.keys())
        if not all([zone in self.vertices.keys() for zone in zones]):
            raise ValueError("zones labels are contained in the kyes in generated tetrahedra.")

        # integrate vertices and tetrahedra in zones
        vertices = self._vertices[zones[0]]
        tetrahedra = self._tetrahedra[zones[0]]
        next_index = vertices.shape[0]
        for zone in zones[1:]:
            vertices = np.concatenate([vertices, self._vertices[zone]], axis=0)
            tetrahedra = np.concatenate([tetrahedra, self._tetrahedra[zone] + next_index], axis=0)
            next_index = vertices.shape[0]

        # cell indices used in this function
        cell_index = []
        for zone in zones:
            cell_index += self._phys_cells[zone]

        # generate Discrete3DMesh instance
        index_func = Discrete3DMesh(vertices, tetrahedra, _sixfold_index_data(cell_index), False, -1)

        # save into path directory using pickel if save is True
        if save is True:
            with open(path, "wb") as f:
                pickle.dump(index_func, f, protocol=pickle.HIGHEST_PROTOCOL)

        return index_func

    def load_index_func(self, path=None):
        path = path or os.path.join(BASE, "data", "emc3_grid.pickel")
        with open(path, "rb") as f:
            return pickle.load(f)

    def plot_zones(self, zone_type=1, toroidal_angle=0.0, dpi=200):

        # load E3E vertices if they are not loaded yet
        if self.r or self.z is None:
            self.load_grids()

        # zones type
        if zone_type == 1:
            zones = ["zone0", "zone1", "zone2", "zone3", "zone4"]
            if toroidal_angle not in self._r["zone0"].keys():
                raise ValueError("toroidal angle is allowed in [0.0, ..., 9.0] in step 0.25 degree.")
        else:
            zones = ["zone11", "zone12", "zone13", "zone14", "zone15"]
            if toroidal_angle not in self._r["zone11"].keys():
                raise ValueError("toroidal angle is allowed in [9.0, ..., 18.0] in step 0.25 degree.")

        # set figure
        fig, ax = plt.subplots(dpi=dpi)

        # plot zone 0 or 11
        for zone in zones:
            if zone in ["zone0", "zone11"]:
                rad_index = self._num_radial[zone] - 1
                vertices = np.array([(self._r[zone][toroidal_angle][rad_index + self._num_radial[zone] * i], self._z[zone][toroidal_angle][rad_index + self._num_radial[zone] * i]) for i in range(self._num_poloidal[zone])] + [(self._r[zone][toroidal_angle][rad_index], self.z[zone][toroidal_angle][rad_index])])
                ax.fill(vertices[:, 0], vertices[:, 1], facecolor='c', edgecolor='k', linewidth=0.0, label=zone)

                # fill center region
                rad_index = 0
                vertices = np.array([(self._r[zone][toroidal_angle][rad_index + self._num_radial[zone] * i], self._z[zone][toroidal_angle][rad_index + self._num_radial[zone] * i]) for i in range(self._num_poloidal[zone])] + [(self._r[zone][toroidal_angle][rad_index], self._z[zone][toroidal_angle][rad_index])])
                ax.fill(vertices[:, 0], vertices[:, 1], facecolor='w', edgecolor='w', linewidth=0.0, label='_nolegend_')

            # plot zone1-4 or 12-15
            else:
                rad_index = self._num_radial[zone] - 1
                vertices = np.array(
                    [(self._r[zone][toroidal_angle][i], self._z[zone][toroidal_angle][i]) for i in range(self._num_radial[zone])]
                    + [(self._r[zone][toroidal_angle][rad_index + self._num_radial[zone] * i], self._z[zone][toroidal_angle][rad_index + self._num_radial[zone] * i]) for i in range(1, self._num_poloidal[zone])]
                    + [(self._r[zone][toroidal_angle][rad_index + self._num_radial[zone] * (self._num_poloidal[zone] - 2) + i],
                        self._z[zone][toroidal_angle][rad_index + self._num_radial[zone] * (self._num_poloidal[zone] - 2) + i]) for i in range(self._num_radial[zone], 0, -1)]
                    + [(self._r[zone][toroidal_angle][0 + self._num_radial[zone] * i], self._z[zone][toroidal_angle][0 + self._num_radial[zone] * i]) for i in range(self._num_poloidal[zone] - 1, -1, -1)]
                )
                ax.fill(vertices[:, 0], vertices[:, 1], edgecolor='k', linewidth=0.0, label=zone)

        ax.set_aspect("equal")
        ax.set_xlabel("R[m]")
        ax.set_ylabel("Z[m]")
        ax.legend()
        ax.set_title(f"$\\phi$ = {toroidal_angle} [deg]", )

        return (fig, ax)

    @property
    def zones(self):
        """Label of EMC3's zones

        Returns
        -------
        list
            zone labels, e.g. :math:`['zone0', 'zone1',...]`
        """
        return self._zones

    @property
    def r(self):
        """r coordinates of EMC3's cell vertices

        Returns
        -------
        dict
            r coordinates of vertices
            ::
              >>> r
              {'zone0': {0.0: [3.593351, 2.559212, ...],
                         0.25: [...],
                         :
                         9.0: [...]},
               'zone1': ...,
                :
               'zone4': ...
               }
        """
        return self._r

    @property
    def z(self):
        return self._z

    @property
    def num_radial(self):
        return self._num_radial

    @property
    def num_poloidal(self):
        return self._num_poloidal

    @property
    def num_toroidal(self):
        return self._num_toroidal

    @property
    def num_cells(self):
        return self._num_cells

    @property
    def vertices(self):
        return self._vertices

    @property
    def tetrahedra(self):
        return self._tetrahedra

    @property
    def phys_cells(self):
        return self._phys_cells


def _sixfold_index_data(index):
    if not isinstance(index, list):
        raise ValueError("index must be list")

    new_index = np.zeros(len(index) * 6, dtype=np.int32)
    j = 0
    for i in index:
        new_index[j:j + 6] = i
        j += 6
    return new_index


if __name__ == '__main__':
    emc = EMC3()
    emc.tetrahedralization(zones=["zone9"])
    emc.generate_index_function(zones=["zone9"])
