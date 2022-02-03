import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import electron_mass, atomic_mass

from raysect.core import translate, Vector3D
from raysect.primitive import Cylinder, Subtract
from raysect.optical.material.emitter.inhomogeneous import NumericalIntegrator

from cherab.openadas import OpenADAS
from cherab.core import Species, Maxwellian, Plasma, Line, elements
from cherab.core.math import Constant3D, ConstantVector3D
from cherab.core.model import ExcitationLine, RecombinationLine, Bremsstrahlung

from .geometry import EMC3
from .cython import EMC3Mapper
from .data_loader import DataLoader

from cherab.lhd.tools.visualization import show_profile_phi_degs


# Const.
RMIN = 2.0  # [m]
RMAX = 5.5
ZMIN = -1.6
ZMAX = 1.6


def import_plasma(parent, species=None):
    """Helper function of generating LHD plasma
    As emissions, H :math:`\\alpha`, H :math:`\\beta`, H :math:`\\gamma`, H :math:`\\delta` are applied.

    Parameters
    ----------
    parent : :obj:`~raysect.core.scenegraph.node.Node`
        Raysect's scene-graph parent node
    species : object , optional
        user-defined species object having composition which is a list of :obj:`~cherab.core.Species` objects
        and electron distribution function attributes,
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
    """Class representing LHD plasma species

    Attributes
    -----------
    electron_distribution : :obj:`~cherab.core.distribution.Maxwellian`
        electron distribution function
    composition : list of :obj:`~cherab.core.Species`
        composition of plasma species, each information of which is
        element, charge, density_distribution, temperature_distribution, bulk_velocity_distribution.
    """

    def __init__(self):

        # load dataloader
        data = DataLoader()

        emc = EMC3()
        func = emc.load_index_func()

        # load data arrays
        e_density = data.density_electron()
        ion_densities = data.density_ions()
        n_densities = data.density_neutrals()
        temp_e_ion = data.temperature_electron_ion()
        temp_n = data.temperature_neutrals()

        bulk_velocity = ConstantVector3D(Vector3D(0, 0, 0))

        # set electron distribution assuming Maxwellian
        self.electron_distribution = Maxwellian(
            EMC3Mapper(func, e_density),
            EMC3Mapper(func, temp_e_ion[0]),
            bulk_velocity,
            electron_mass
        )

        # initialize composition
        self.composition = []

        # append species to composition list
        # H
        self.set_species(
            element="hydrogen",
            charge=0,
            density=EMC3Mapper(func, n_densities["H"]),
            temperature=EMC3Mapper(func, temp_n["H"]),
        )
        # H+
        self.set_species(
            element="hydrogen",
            charge=1,
            density=EMC3Mapper(func, ion_densities["H+"]),
            temperature=EMC3Mapper(func, temp_e_ion[1]),
        )
        # C1+ - C6+
        for i in range(1, 7):
            self.set_species(
                element="carbon",
                charge=i,
                density=EMC3Mapper(func, ion_densities[f"C{i}+"]),
                temperature=EMC3Mapper(func, temp_e_ion[1]),
            )

        # Ne1+ - Ne10+
        for i in range(1, 11):
            self.set_species(
                element="neon",
                charge=i,
                density=EMC3Mapper(func, ion_densities[f"Ne{i}+"]),
                temperature=EMC3Mapper(func, temp_e_ion[1]),
            )

    def __repr__(self):
        return f"{self.composition}"

    def set_species(
        self,
        element=None,
        charge=0,
        density=Constant3D(1.0e19),
        temperature=Constant3D(1.0e2),
        bulk_velocity=ConstantVector3D(Vector3D(0, 0, 0)),
    ):
        """add species to composition which is assumed to be Maxwellian distribution.

        Parameters
        ------------
        element : str, required
            element name registored in cherabs elements.pyx, by default None
        charge : int, required
            element's charge state, by default 0
        density : :obj:`~raysect.core.math.function.float.function3d.base.Function3D`, optional
            density distribution, by default Constant3D(1.0e19)
        temperature : :obj:`~raysect.core.math.function.float.function3d.base.Function3D`, optional
            temperature distribution, by default Constant3D(1.0e2)
        bulk_velocity : :obj:`~raysect.core.math.function.vector3d.function3d.base.Function3D`, optional
            bulk velocity, by default ConstantVector3D(0)
        """

        if not element:
            message = "Parameter 'element' is required to be input."
            raise ValueError(message)

        # extract specified element object
        element = getattr(elements, element, None)
        if not element:
            message = (
                f"element name '{element}' is not implemented."
                f"You can implement manually using Element class"
            )
            raise NotImplementedError(message)

        # element mass
        element_mass = element.atomic_weight * atomic_mass

        # Maxwellian distribution
        distribution = Maxwellian(density, temperature, bulk_velocity, element_mass)

        # append plasma.composition
        self.composition.append(Species(element, charge, distribution))

    def plot_distribution(self, res=5.0e-3):
        """plot species density and temperature profile

        Parameters
        ----------
        res : float, optional
            Spactial resolution for sampling, by default 0.005 [m]
        """
        # plot electron distribution
        fig, _ = show_profile_phi_degs(
            self.electron_distribution._density, masked=True, clabel="density [1/m$^3$]", resolution=res
        )
        fig.suptitle("electron density", y=0.92)

        fig, _ = show_profile_phi_degs(
            self.electron_distribution._temperature, masked=True, clabel="temperature [eV]", resolution=res
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
                ["density [1/m$^3$]", "temperature [eV]"]
            ):
                fig, _ = show_profile_phi_degs(
                    func, masked=True, clabel=clabel
                )
                fig.suptitle(title, y=0.92)

        plt.show()


# For debugging
if __name__ == "__main__":
    from raysect.core import World

    species = LHDSpecies()
    species.plot_distribution()

    world = World()
    plasma = import_plasma()
    print([i for i in plasma.models])
    pass
