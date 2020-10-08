# -*- coding: utf-8 -*-
from __future__ import print_function
import os
from raysect.optical import rotate
from raysect.optical.library import RoughBeryllium, RoughIron, RoughTungsten
from raysect.optical.material import AbsorbingSurface
from raysect.primitive.mesh import Mesh


def load_pfc_mesh(
    world,
    path=os.path.join(os.path.dirname(__file__), "geometry", "data", "RSMfiles"),
    reflections=True,
    roughness={"Be": 0.26, "W": 0.29, "Ss": 0.13},
):
    """ Loads LHD PFC mesh and connects it to Raysect World() instance.
        Note that currently the entire first wall is obtained by copying and rotating the 1st sector 5 times.
        The coordinates of custom port plugs are given in TGCS.
        The coordinates of general port plugs are given in internal coordinate system (which can be
        transformed to TGCS for the sector 1 by rotating on 10 deg counter-clockwise over Z-axis).

        Parameters
        ----------
        world : object instance
            Raysect World() instance
        path : string, optional
            Path to .rsm mesh files
        reflections : bool, optional
            Reflection on/off. Default: True.
        roughness : dict, optional
            Roughness dict for PFC materials ('Be', 'W', 'Ss').
        """

    pfc_list = ["vessel", "plates"]

    # How many times each PFC element must be copy-pasted
    ncopy = {"vessel": 1, "plates": 5}

    if reflections:
        materials = {"vessel": RoughBeryllium(roughness["Be"]), "plates": RoughBeryllium(roughness["Be"])}
    else:
        materials = {"vessel": AbsorbingSurface(), "plates": AbsorbingSurface()}

    mesh = {}

    for pfc in pfc_list:
        mesh[pfc] = [
            Mesh.from_file(os.path.join(path, f"{pfc}.rsm"), parent=world, material=materials[pfc])
        ]  # master element
        angle = 360.0 / ncopy[pfc]  # rotate around Z by this angle
        for i in range(1, ncopy[pfc]):  # copies of the master element
            mesh[pfc].append(
                mesh[pfc][0].instance(parent=world, transform=rotate(0, 0, angle * i), material=materials[pfc])
            )

    return mesh


if __name__ == "__main__":
    pass
