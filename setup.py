from __future__ import annotations

import multiprocessing
import os
import os.path as path
import sys

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

multiprocessing.set_start_method("fork")
NCORES = multiprocessing.cpu_count()

force = False
profile = False

if "--force" in sys.argv:
    force = True
    del sys.argv[sys.argv.index("--force")]

if "--profile" in sys.argv:
    profile = True
    del sys.argv[sys.argv.index("--profile")]


compilation_includes = [".", numpy.get_include()]
compilation_args = ["-fopenmp"]
compilation_links = ["-fopenmp"]
cython_directives = {"language_level": 3}

setup_path = path.dirname(path.abspath(__file__))

# build .pyx extension list
EXTENSIONS_TO_BUILD = []
for root, dirs, files in os.walk(setup_path):
    for file in files:
        if path.splitext(file)[1] == ".pyx":
            pyx_file = path.relpath(path.join(root, file), setup_path)
            module = path.splitext(pyx_file)[0].replace("/", ".")
            EXTENSIONS_TO_BUILD.append(
                Extension(
                    module,
                    [pyx_file],
                    include_dirs=compilation_includes,
                    extra_compile_args=compilation_args,
                    extra_link_args=compilation_links,
                )
            )

if profile:
    cython_directives["profile"] = True

# generate .c files from .pyx
EXTENSIONS = cythonize(
    EXTENSIONS_TO_BUILD,
    nthreads=NCORES,
    force=force,
    compiler_directives=cython_directives,
)


def setup_given_extensions(extensions: list[Extension]):
    setup(
        ext_modules=extensions,
    )


def setup_extensions_in_parallel():
    mp = multiprocessing.get_context("fork")
    pool = mp.Pool(processes=NCORES)
    extensions = [[e] for e in EXTENSIONS]
    pool.imap(setup_given_extensions, extensions)
    pool.close()
    pool.join()


if __name__ == "__main__":
    setup_extensions_in_parallel()
