"""Module offers helper functions to generate :obj:`~cherab.core.Plasma`
object."""
from __future__ import annotations

import h5py
from cherab.core import Line, Maxwellian, Plasma, Species, elements
from cherab.core.math import Constant3D, ConstantVector3D
from cherab.core.model import Bremsstrahlung, ExcitationLine, RecombinationLine
from cherab.openadas import OpenADAS
from matplotlib import pyplot as plt
from raysect.core import Node, Vector3D, translate
from raysect.core.math.function.float.function3d.base import Function3D
from raysect.core.math.function.vector3d.function3d.base import Function3D as VectorFunction3D
from raysect.optical.material.emitter.inhomogeneous import NumericalIntegrator
from raysect.primitive import Cylinder, Subtract
from scipy.constants import atomic_mass, electron_mass

from ..tools import Spinner
from ..tools.visualization import show_profile_phi_degs
from .cython import EMC3Mapper
from .geometry import PhysIndex
from .repository.utility import DEFAULT_HDF5_PATH

__all__ = ["import_plasma", "LHDSpecies"]


# Const.
RMIN = 2.0  # [m]
RMAX = 5.5
ZMIN = -1.6
ZMAX = 1.6

# Default Distribution Function
DENSITY = Constant3D(1.0e19)  # [1/m^3]
TEMPERATURE = Constant3D(1.0e2)  # [eV]
BULK_V = ConstantVector3D(Vector3D(0, 0, 0))


@Spinner(text="Loading Plasma Object...", timer=True)
def import_plasma(parent: Node, species: Species | None = None) -> Plasma:
    """Helper function of generating LHD plasma As emissions, H
    :math:`\\alpha`, H :math:`\\beta`, H :math:`\\gamma`, H :math:`\\delta` are
    applied.

    Parameters
    ----------
    parent
        Raysect's scene-graph parent node
    species
        user-defined species object having composition which is a list of
        :obj:`~cherab.core.Species` objects and electron distribution function attributes,
        by default :py:class:`.LHDSpecies`

    Returns
    -------
    :obj:`~cherab.core.Plasma`
        plasma object
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
    species = species or LHDSpecies()
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

    Attributes
    -----------
    electron_distribution : :obj:`~cherab.core.distribution.Maxwellian`
        electron distribution function
    composition : list of :obj:`~cherab.core.Species`
        composition of plasma species, each information of which is
        element, charge, density_distribution, temperature_distribution, bulk_velocity_distribution.
    """

    def __init__(self):
        # Load dataset from HDF5 file
        with h5py.File(DEFAULT_HDF5_PATH, mode="r") as h5file:

            # load index function
            func = PhysIndex()

            # data group
            data_group = h5file["grid-360/data/"]

            bulk_velocity = ConstantVector3D(Vector3D(0, 0, 0))

            # set electron distribution assuming Maxwellian
            self.electron_distribution = Maxwellian(
                EMC3Mapper(func, data_group["density/electron"][:]),
                EMC3Mapper(func, data_group["temperature/electron"][:]),
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
                density=EMC3Mapper(func, data_group["density/H"]),
                temperature=EMC3Mapper(func, data_group["temperature/H"]),
            )
            # H+
            self.set_species(
                "hydrogen",
                1,
                density=EMC3Mapper(func, data_group["density/H+"]),
                temperature=EMC3Mapper(func, data_group["temperature/ion"]),
            )
            # C1+ - C6+
            for i in range(1, 7):
                self.set_species(
                    "carbon",
                    i,
                    density=EMC3Mapper(func, data_group[f"density/C{i}+"]),
                    temperature=EMC3Mapper(func, data_group["temperature/ion"]),
                )

            # Ne1+ - Ne10+
            for i in range(1, 11):
                self.set_species(
                    "neon",
                    i,
                    density=EMC3Mapper(func, data_group[f"density/Ne{i}+"]),
                    temperature=EMC3Mapper(func, data_group["temperature/ion"]),
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
        """add species to composition which is assumed to be Maxwellian
        distribution.

        Parameters
        ----------
        element
            element name registored in cherabs elements.pyx
        charge
            element's charge state, by default 0
        density
            density distribution, by default :obj:`~cherab.core.math.Constant3D` (1.0e19)
        temperature
            temperature distribution, by default :obj:`~cherab.core.math.Constant3D` (1.0e2)
        bulk_velocity
            bulk velocity, by default :obj:`~cherab.core.math.ConstantVector3D` (0)
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
        """plot species density and temperature profile.

        Parameters
        ----------
        res
            Spactial resolution for sampling, by default 0.005 [m]
        """
        # plot electron distribution
        fig, _ = show_profile_phi_degs(
            self.electron_distribution._density,
            masked="wall",
            clabel="density [1/m$^3$]",
            resolution=res,
        )
        fig.suptitle("electron density", y=0.92)

        fig, _ = show_profile_phi_degs(
            self.electron_distribution._temperature,
            masked="wall",
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
                fig, _ = show_profile_phi_degs(func, masked="wall", clabel=clabel)
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
