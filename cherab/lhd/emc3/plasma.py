"""Module offers helper functions to generate `~cherab.core.Plasma` object."""

from __future__ import annotations

import h5py  # noqa: F401
import xarray as xr
from matplotlib import pyplot as plt
from raysect.core import Vector3D, translate
from raysect.core.math.function.float.function3d.base import Function3D
from raysect.core.math.function.vector3d.function3d.base import Function3D as VectorFunction3D
from raysect.core.scenegraph._nodebase import _NodeBase
from raysect.optical.material.emitter.inhomogeneous import NumericalIntegrator
from raysect.primitive import Cylinder, Subtract
from scipy.constants import atomic_mass, electron_mass

from cherab.core import Line, Maxwellian, Plasma, Species, elements
from cherab.core.math import Constant3D, ConstantVector3D
from cherab.core.model import Bremsstrahlung, ExcitationLine, RecombinationLine
from cherab.openadas import OpenADAS

from ..tools import Spinner
from ..tools.fetch import fetch_file
from ..tools.visualization import show_profile_phi_degs
from .cython import Mapper
from .indices import load_index_func

__all__ = ["import_plasma", "LHDSpecies"]


# Const.
RMIN = 2.0  # [m]
RMAX = 5.5
ZMIN = -1.6
ZMAX = 1.6

# Default Zones
ZONES = [
    "zone0",
    "zone1",
    "zone2",
    "zone3",
    "zone4",
    "zone11",
    "zone12",
    "zone13",
    "zone14",
    "zone15",
]

# Default Distribution Function
DENSITY = Constant3D(1.0e19)  # [1/m^3]
TEMPERATURE = Constant3D(1.0e2)  # [eV]
BULK_V = ConstantVector3D(Vector3D(0, 0, 0))


@Spinner(text="Loading Plasma Object...", timer=True)
def import_plasma(
    parent: _NodeBase,
    species: Species | None = None,
    zones: list[str] = ZONES,
    dataset: str = "emc3/grid-360.nc",
    **kwargs,
) -> Plasma:
    """Helper function of generating LHD plasma.

    As emissions, Hɑ, Hβ, Hγ, Hδ are applied.

    Parameters
    ----------
    parent : `_NodeBase`
        Parent node of this plasma in the scenegraph, often `~raysect.core.World` object.
    species : `~cherab.core.Species`, optional
        User-defined species object having composition which is a list of `~cherab.core.Species`
        objects and electron distribution function attributes, by default `.LHDSpecies`.
    zones : list[str], optional
        List of zone names, by default `["zone0", "zone1", "zone2", "zone3", "zone4", "zone11", "zone12", "zone13", "zone14", "zone15"]`.
    dataset : str, optional
        Name of dataset, by default ``"emc3/grid-360.nc"``.

    Returns
    -------
    `~cherab.core.Plasma`
        Plasma object.
    """
    # create atomic data source
    adas = OpenADAS(permit_extrapolation=True)

    # generate plasma object instance
    plasma = Plasma(parent=parent, name="LHD_plasma")

    # setting plasma properties
    plasma.atomic_data = adas
    plasma.integrator = NumericalIntegrator(step=0.001)
    plasma.b_field = ConstantVector3D(Vector3D(0, 0, 0))

    # create plasma geometry as subtraction of two cylinders
    inner_radius = RMIN
    outer_radius = RMAX
    height = ZMAX - ZMIN

    inner_cylinder = Cylinder(inner_radius, height)
    outer_cylinder = Cylinder(outer_radius, height)

    plasma.geometry = Subtract(outer_cylinder, inner_cylinder)
    plasma.geometry_transform = translate(0, 0, ZMIN)

    # apply species to plasma
    species = species or LHDSpecies(zones=zones, dataset=dataset, **kwargs)
    plasma.composition = species.composition
    plasma.electron_distribution = species.electron_distribution

    # apply emission from plasma
    h_alpha = Line(elements.hydrogen, 0, (3, 2))  # , wavelength=656.279)
    h_beta = Line(elements.hydrogen, 0, (4, 2))  # , wavelength=486.135)
    h_gamma = Line(elements.hydrogen, 0, (5, 2))  # , wavelength=434.0472)
    h_delta = Line(elements.hydrogen, 0, (6, 2))  # , wavelength=410.1734)
    # ciii_777 = Line(
    #     elements.carbon, 2, ("1s2 2p(2P°) 3d 1D°", " 1s2 2p(2P°) 3p  1P")
    # )  # , wavelength=770.743)
    plasma.models = [
        Bremsstrahlung(),
        ExcitationLine(h_alpha),
        ExcitationLine(h_beta),
        ExcitationLine(h_gamma),
        ExcitationLine(h_delta),
        # ExcitationLine(ciii_777),
        RecombinationLine(h_alpha),
        RecombinationLine(h_beta),
        RecombinationLine(h_gamma),
        RecombinationLine(h_delta),
        # RecombinationLine(ciii_777),
    ]

    return plasma


class LHDSpecies:
    """Class representing LHD plasma species.

    Parameters
    ----------
    zones : list[str], optional
        List of zone names, by default `["zone0", "zone1", "zone2", "zone3", "zone4", "zone11", "zone12", "zone13", "zone14", "zone15"]`.
    dataset : str, optional
        Name of dataset, by default ``"emc3/grid-360.nc"``.
    **kwargs
        Keyword arguments for `.fetch_file`.

    Attributes
    ----------
    electron_distribution : `~cherab.core.distribution.Maxwellian`
        Electron distribution function.
    composition : list[`~cherab.core.Species`]
        Composition of plasma species, each information of which is element, charge,
        density_distribution, temperature_distribution, bulk_velocity_distribution.
    """

    def __init__(
        self, zones: list[str] = ZONES, dataset: str = "emc3/grid-360.nc", **kwargs
    ) -> None:
        path = fetch_file(dataset, **kwargs)

        # Load index functions
        funcs = []
        for zone in zones:
            func, _ = load_index_func(zone, index_type="physics", dataset=dataset)
            funcs.append(func)

        funcs = sum(funcs)

        # data group
        data_tree = xr.open_datatree(path, group="data")
        bulk_velocity = ConstantVector3D(Vector3D(0, 0, 0))
        # set electron distribution assuming Maxwellian
        self.electron_distribution = Maxwellian(
            Mapper(funcs, data_tree["density/electron"].data),
            Mapper(funcs, data_tree["temperature/electron"].data),
            bulk_velocity,
            electron_mass,
        )
        # initialize composition
        self.composition = []
        # append species to composition list
        # H
        self.set_species(
            "hydrogen",
            0,
            density=Mapper(funcs, data_tree["density/H"].data),
            temperature=Mapper(funcs, data_tree["temperature/H"].data),
        )

        # H+
        self.set_species(
            "hydrogen",
            1,
            density=Mapper(funcs, data_tree["density/H+"].data),
            temperature=Mapper(funcs, data_tree["temperature/ion"].data),
        )

        # C1+ - C6+
        for i in range(1, 7):
            self.set_species(
                "carbon",
                i,
                density=Mapper(funcs, data_tree[f"density/C{i}+"]),
                temperature=Mapper(funcs, data_tree["temperature/ion"]),
            )

        # Ne1+ - Ne10+
        for i in range(1, 11):
            self.set_species(
                "neon",
                i,
                density=Mapper(funcs, data_tree[f"density/Ne{i}+"]),
                temperature=Mapper(funcs, data_tree["temperature/ion"]),
            )

    def __repr__(self):
        return f"{self.composition}"

    def set_species(
        self,
        element: str,
        charge: int,
        density: Function3D = DENSITY,
        temperature: Function3D = TEMPERATURE,
        bulk_velocity: VectorFunction3D = BULK_V,
    ) -> None:
        """Add species to composition which is assumed to be Maxwellian distribution.

        Parameters
        ----------
        element : str
            Element name registered in cherabs `elements.pyx` module.
        charge : int
            Element's charge state, by default 0.
        density : `~cherab.core.math.Function3D`, optional
            Density distribution, by default `~cherab.core.math.Constant3D` (1.0e19).
        temperature : `~cherab.core.math.Function3D`, optional
            Temperature distribution, by default `~cherab.core.math.Constant3D` (1.0e2).
        bulk_velocity : `~cherab.core.math.VectorFunction3D`, optional
            Bulk velocity, by default `~cherab.core.math.ConstantVector3D` (0).
        """
        # extract specified element object
        element_obj = getattr(elements, element, None)
        if not element_obj:
            message = (
                f"element name '{element}' is not implemented."
                f"You can implement manually using Element class"
            )
            raise NotImplementedError(message)

        # element mass
        element_mass = element_obj.atomic_weight * atomic_mass

        # Maxwellian distribution
        distribution = Maxwellian(density, temperature, bulk_velocity, element_mass)

        # append plasma.composition
        self.composition.append(Species(element_obj, charge, distribution))

    def plot_distribution(self, res: float = 5.0e-3):
        """Plot species density and temperature profile.

        Parameters
        ----------
        res : float, optional
            Spactial resolution for sampling, by default 0.005 [m].
        """
        # plot electron distribution
        fig, _ = show_profile_phi_degs(
            self.electron_distribution._density,
            mask="wall",
            clabel="density [1/m$^3$]",
            resolution=res,
        )
        fig.suptitle("electron density", y=0.92)

        fig, _ = show_profile_phi_degs(
            self.electron_distribution._temperature,
            mask="wall",
            clabel="temperature [eV]",
            resolution=res,
        )
        fig.suptitle("electron temperature", y=0.92)

        # species sampling
        for species in self.composition:
            # plot
            for func, title, clabel in zip(
                [species.distribution._density, species.distribution._temperature],
                [
                    f"{species.element.symbol}{species.charge}+ density",
                    f"{species.element.symbol}{species.charge}+ temperature",
                ],
                ["density [1/m$^3$]", "temperature [eV]"],
                strict=True,
            ):
                fig, _ = show_profile_phi_degs(func, mask="wall", clabel=clabel)
                fig.suptitle(title, y=0.92)

        plt.show()


# For debugging
if __name__ == "__main__":
    from raysect.core import World

    species = LHDSpecies()
    species.plot_distribution()

    world = World()
    plasma = import_plasma(world)
    print([i for i in plasma.models])
    pass
