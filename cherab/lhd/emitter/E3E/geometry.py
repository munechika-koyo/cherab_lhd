import os
import numpy as np
import pickle
from matplotlib import pyplot as plt
from cherab.lhd.emitter.E3E.cython import Discrete3DMesh

BASE = os.path.dirname(__file__)
GRID_PATH = os.path.join(BASE, "data", "grid-360.pickle")
CELLGEO_PATH = os.path.join(BASE, "data", "CELL_GEO.pickle")
INDEX_FUNC_PATH = os.path.join(BASE, "data", "emc3_grid.pickle")


class EMC3:
    """EMC3-EIRINE class
    This class is implemented for EMC3's grids, physical cell indices, and tetrahedralization
    to enable raysect to handle them as an emitter.
    """

    def __init__(self) -> None:
        # initialize properties
        self._zones = [f"zone{i}" for i in [j for j in range(0, 4 + 1)] + [k for k in range(11, 15 + 1)]]
        self._grids = {}
        self._num_radial = {}
        self._num_poloidal = {}
        self._num_toroidal = {}
        self._num_cells = {}
        self._vertices = {}
        self._tetrahedra = {}
        self._phys_cells = {}

    def load_grids(self, path=None) -> None:
        """Load EMC3-Eirine grids (r, z, phi) coordinates, which is classified by zones.
        The grids are loaded as the type of dictonary.
        Labels of zones are in ['zone0',..., 'zone21'].
        The number of radial, poloindal toroidal and cells are also loaded.

        Parameters
        ----------
        path : str, optional
            path to the grids text file, by default ".../data/grid-360.pickle"
        """

        # path to E3E grid file
        self.grid_path = path or GRID_PATH

        # load geometry cell's grids and numbers
        with open(self.grid_path, "rb") as f:
            self._grids, self._num_radial, self._num_poloidal, self._num_toroidal, self._num_cells = pickle.load(f)

    def load_cell_index(self, path=None) -> None:
        """Load cell geometry indices
        EMC3 has numerous geometric cells, the number of which is defined :math:`num_cells` in each zones.
        However, EMC3 calculates the physical values such as plasma density in each cell combined several geometric cells,
        which is called "physical" cells. Their relationship between geomtric and physical cells indices is written in CELL_GEO file.
        In this method, the pickled file "CELL_GEO.pickle" is loaded by default.

        Parameters
        ----------
        path : str, optional
            path to the CELL_GEO.pickle file, by default ".../data/CELL_GEO.pickle"
            The default file "CELL_GEO.pickle" has geometric indices in all zones.
        """
        # path to CELL_GEO file
        self.cell_index_path = path or CELLGEO_PATH

        # load cell_geo indices
        with open(self.cell_index_path, "rb") as f:
            cell_index = pickle.load(f)

        # classify all cell_geo indices by zone labels
        if not self._num_cells:
            self.load_grids()

        start = 0
        for zone in self._num_cells.keys():
            num = self._num_cells[zone]
            self._phys_cells[zone] = cell_index[start:start + num]
            start += num

    def tetrahedralization(self, zones=None) -> None:
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
        if not self.grids:
            self.load_grids()

        # check if all elements of zones arg is included in zones dict.keys.
        if not all([zone in self.num_cells.keys() for zone in zones]):
            raise ValueError("zones labels are contained in the property 'zones'.")

        # initialize local variables
        vertices = {}
        tetrahedra = {}

        for (zone, grids), num_rad, num_pol, num_tor in zip(self.grids.items(), self.num_radial.values(), self.num_poloidal.values(), self.num_toroidal.values()):

            if zone not in zones:
                continue

            # set local variables: vertices and tetrahedral indices in each zones
            num_total = num_rad * num_pol
            vertices[zone] = np.zeros((num_total * num_tor, 3), dtype=np.float64)
            tetrahedra[zone] = np.zeros((6 * self._num_cells[zone], 4), dtype=np.uint32)

            # gengerate vertices
            start = 0
            for i_phi in range(num_tor):
                phi = np.deg2rad(grids[0, 2, i_phi])
                vertices[zone][start: start + num_total, :] = np.vstack([grids[:, 0, i_phi] * np.cos(phi),
                                                                         grids[:, 0, i_phi] * np.sin(phi),
                                                                         grids[:, 1, i_phi]]).T
                start += num_total

            # geomtric cell indices at one toroidal plane
            faces = self.generate_faces(zone=zone)

            # Divide a cell into six tetrahedra.
            tet_id = 0
            offset = 0
            for i in range(num_tor - 1):
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

    def generate_index_function(self, zones=None, save=True, path=None) -> Discrete3DMesh:
        """Generate EMC3's Physical Index function
        and picklize it to save.

        Parameters
        ----------
        zones : list, optional
            zones label, by default self.zones :math:`['zone0', 'zone1', ...]`
        save : bool, optional
            whether or not to store this function, by default True
        path : str, optional
            path to saving file name, by default "../data/emc3_grid.pickle"

        Returns
        -------
        IntegerFunction3D
            cythonized function returing a EMC3's physical indix corresponding to
            (x, y, z) coords.
        """

        # path to save function as pickel
        path = path or INDEX_FUNC_PATH

        # choose zones
        if not self._vertices:
            raise ValueError("vertices attribution doesn't exist. The tetrahedralization method must be excuted before calling.")

        zones = zones or list(self._vertices.keys())
        if not all([zone in self._vertices.keys() for zone in zones]):
            raise ValueError("zones labels are contained in the keys in generated tetrahedra.")

        # load CELL_GEO file if yet
        if not self._phys_cells:
            self.load_cell_index()

        # integrate vertices and tetrahedra in zones
        vertices = self._vertices[zones[0]]
        tetrahedra = self._tetrahedra[zones[0]]
        cell_index = self._phys_cells[zones[0]]
        next_index = vertices.shape[0]
        for zone in zones[1:]:
            vertices = np.concatenate([vertices, self._vertices[zone]], axis=0)
            tetrahedra = np.concatenate([tetrahedra, self._tetrahedra[zone] + next_index], axis=0)
            cell_index = np.concatenate([cell_index, self._phys_cells[zone]], axis=0)
            next_index = vertices.shape[0]

        # generate Discrete3DMesh instance
        index_func = Discrete3DMesh(vertices, tetrahedra, EMC3.sixfold_data(cell_index), False, -1)

        # save into path directory using pickel if save is True
        if save:
            if os.path.isdir(os.path.split(path)[1]):
                raise ValueError("the endpoint of path must be filename not directory")
            if os.path.splitext(path)[1] != ".pickle":
                path += ".pickle"
            with open(path, "wb") as f:
                pickle.dump(index_func, f, protocol=pickle.HIGHEST_PROTOCOL)

        return index_func

    def load_index_func(self, path=None):
        """Load pickeled EMC3's physical index function

        Parameters
        ----------
        path : str, optional
            path to pickeld file, by default "../data/emc3_grid.pickle"

        Returns
        -------
        IntegerFunction3D
            cythonized function returing a EMC3's physical index corresponding to
            (x, y, z) coords.
        """
        path = path or INDEX_FUNC_PATH
        with open(path, "rb") as f:
            return pickle.load(f)

    def generate_faces(self, zone="zone0") -> list:
        """generate cell indices

        Parameters
        ----------
        zone : str, optional
            label of zones, by default "zone0"

        Returns
        -------
        list
            containing face indices

        ::
            >>> emc = EMC3()
            >>> faces = emc.generate_faces(zone="zone0")
            >>> faces
            [(0, 1, 82, 81),
            (1, 2, 83, 82),
            (2, 3, 84, 83),
            ...,
            (48598, 48599, 48680, 48679)]
        """
        faces = []
        start = 0
        if not self._grids:
            self.load_grids()
        N_rad = self._num_radial[zone]
        N_pol = self._num_poloidal[zone]

        for start in range(0, N_rad * (N_pol - 1), N_rad):
            for i in range(start, start + N_rad - 1):
                faces += [(i, i + 1, i + 1 + N_rad, i + N_rad)]

        return faces

    def plot_zones(self, zone_type=1, toroidal_angle=0.0, dpi=200):
        """plot EMC3's zones in r - z plane
        EMC3 grids are classified into 2 types: (type1) 0 - 9 toridal degree,
                                                (type2) 9 - 18 toroidal degree.
        Domains of each zone (zone0 - 4, zone11-15) are colored.

        Parameters
        ----------
        zone_type : int, optional
            select either type1 or type2 zone, by default 1
        toroidal_angle : float, optional
            toroidal angle [degree].
            each type of zones has classified as follows:
                type1: 0, 0.25, ... 9.0 degree and
                type2: 9.0 9.25, ..., 18.0 degree, by default 0.0
        dpi : int, optional
            figure dpi, by default 200

        Returns
        -------
        (fig, ax): tuple
            matplotlib.Figure & Axes objects
        """

        # load E3E grids if they are not loaded yet
        if not self.grids:
            self.load_grids()

        # zones type
        if zone_type == 1:
            zones = ["zone0", "zone1", "zone2", "zone3", "zone4"]
            if toroidal_angle not in self.grids["zone0"][0, 2, :]:
                raise ValueError("toroidal angle is allowed in [0.0, ..., 9.0] in step 0.25 degree.")
        else:
            zones = ["zone11", "zone12", "zone13", "zone14", "zone15"]
            if toroidal_angle not in self.grids["zone11"][0, 2, :]:
                raise ValueError("toroidal angle is allowed in [9.0, ..., 18.0] in step 0.25 degree.")

        # search for the toroidal index in self.grids corresponding to toroidal_angle
        i_phi = np.where(self.grids[zones[0]][0, 2, :] == toroidal_angle)[0][0]

        # set figure
        fig, ax = plt.subplots(dpi=dpi)

        # plot zone 0 or 11
        for zone in zones:
            N_rad = self._num_radial[zone]
            N_pol = self._num_poloidal[zone]
            if zone in ["zone0", "zone11"]:
                vertices = self.grids[zone][N_rad - 1::N_rad, 0:2, i_phi]
                ax.fill(vertices[:, 0], vertices[:, 1], facecolor='c', edgecolor='k', linewidth=0.0, label=zone)

                # fill center region
                vertices = self.grids[zone][0::N_rad, 0:2, i_phi]
                ax.fill(vertices[:, 0], vertices[:, 1], facecolor='w', edgecolor='w', linewidth=0.0, label='_nolegend_')

            # plot zone0-4 or 11-15
            else:
                vertices = np.vstack((
                    self.grids[zone][0:N_rad - 1, 0:2, i_phi],
                    self.grids[zone][N_rad - 1::N_rad, 0:2, i_phi],
                    self.grids[zone][N_rad * (N_pol - 1) + 1:-1, 0:2, i_phi][::-1],
                    self.grids[zone][0::N_rad, 0:2, i_phi][::-1],
                ))
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
    def grids(self):
        """ (r, z, phi) coordinates of EMC3's cell vertices.
        This property value is obtained after calling load_grids method.

        Returns
        -------
        dict
            numpy.ndarray (r, z, phi) coordinates of vertices
            ::
              >>> grids
              {'zone0': array([[[ 3.593351e+00,  3.593307e+00,  3.593176e+00, ...,
                                  3.551275e+00,  3.549266e+00,  3.547254e+00],
                                [-0.000000e+00, -1.835000e-03, -3.667000e-03, ...,
                                 -4.099100e-02, -4.103600e-02, -4.099800e-02],
                                [ 0.000000e+00,  2.500000e-01,  5.000000e-01, ...,
                                  8.500000e+00,  8.750000e+00,  9.000000e+00]],
                                  ...
                               [[ 3.262600e+00,  3.262447e+00,  3.261987e+00, ...,
                                  3.096423e+00,  3.087519e+00,  3.078508e+00],
                                [ 0.000000e+00, -4.002000e-03, -7.995000e-03, ...,
                                 -7.012100e-02, -6.796400e-02, -6.543900e-02],
                                [ 0.000000e+00,  2.500000e-01,  5.000000e-01, ...,
                                  8.500000e+00,  8.750000e+00,  9.000000e+00]]]),
               'zone1': ...,
                :
               'zone21': ...
               }
        """
        return self._grids

    @property
    def num_radial(self):
        """Number of cell vertices in radial direction.
        This property value is obtained after calling load_grids method.

        Returns
        -------
        dict
            key: zone label, value: int number
            ::
              >>> num_radial
              {'zone0': 81,
               'zone1': 97,
                :
               'zone21': 97
               }
        """
        return self._num_radial

    @property
    def num_poloidal(self):
        """Number of cell vertices in poloidal direction.
        This property value is obtained after calling load_grids method.

        Returns
        -------
        dict
            key: zone label, value: int number
            ::
              >>> num_poloidal
              {'zone0': 600,
               'zone1': 77,
                :
               'zone21': 9
               }
        """
        return self._num_poloidal

    @property
    def num_toroidal(self):
        """Number of cell vertices in toroidal direction.
        This property value is obtained after calling load_grids method.

        Returns
        -------
        dict
            key: zone label, value: int number
            ::
              >>> num_toroidal
              {'zone0': 37,
               'zone1': 37,
                :
               'zone21': 37
               }
        """
        return self._num_toroidal

    @property
    def num_cells(self):
        """Number of geometric cells in EMC3-Eirene in each zones.
        This property value is obtained after calling load_grids method.

        Returns
        -------
        dict
            key: zone label, value: int number
            ::
              >>> num_cells
              {'zone0': 1728000,
               'zone1': 262656,
                :
               'zone21': 27648
               }
        """
        return self._num_cells

    @property
    def vertices(self):
        """EMC3's grids vertex coordinates in regular 3D (X, Y, Z).
        This property value is obtained after calling tetrahedralization method.

        Returns
        -------
        dict
            key: zone label, value: numpy.ndarray
            ::
              >>> vertices
              {'zone1': array([[3.718866  , 0.        , 1.209099  ],
                               [3.718211  , 0.        , 1.209545  ],
                               ...
                               [3.82106001, 0.60519645, 1.599694  ]]),
               'zone2':...
               }
        """
        return self._vertices

    @property
    def tetrahedra(self):
        """EMC3's grids vertex indices constituting tetrahedral meshes.
        This property value is obtained after calling tetrahedralization method.

        Returns
        -------
        dict
            key: zone label, value: numpy.ndarray
            ::
              >>> tetrahedra
              {'zone1': array([[  7469,      0,      1,     98],
                               [  7566,     97,      0,     98],
                               ...
                               [268786, 268883, 276254, 276352]], dtype=uint32)),
               'zone2':...
               }
        """
        return self._tetrahedra

    @property
    def phys_cells(self):
        """EMC3's Physical cell indices
        This property value is obtained after calling load_cell_index method.

        Returns
        -------
        dict
            key: zone label, value: numpy.array
            ::
              >>> phys_cells
              {'zone0': array([314722, 314722, 314722, 314722, 314770, ...], dtype=uint32),
               'zone1': array([...]),
                :
               'zone21':array([...])
               }
        """
        return self._phys_cells

    @staticmethod
    def sixfold_data(data) -> np.ndarray:
        """increse given data six-fold.

        Parameters
        ----------
        data : 1D array-like


        Returns
        -------
        numpy.ndarray

        ::
            >>> a = [0, 1, 2]
            >>> b = EMC3.sixfold_data(a)
            >>> b
            array([0., 0., 0., 0., 0., 0.,
                   1., 1., 1., 1., 1., 1.,
                   2., 2., 2., 2., 2., 2.])

        """
        data = np.asarray_chkfinite(data)
        if data.ndim != 1:
            raise ValueError("data must be an 1D array.")

        new_data = np.zeros(data.shape[0] * 6)
        i = 0
        for value in data:
            new_data[i:i + 6] = value
            i += 6
        return new_data


if __name__ == '__main__':
    emc = EMC3()
    emc.load_grids()
    # emc.tetrahedralization()
    # emc.generate_index_function()
    pass
