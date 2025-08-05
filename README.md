# CHERAB-LHD

|         |                                                                                                                       |
| ------- | --------------------------------------------------------------------------------------------------------------------- |
| CI/CD   | [![CI][CI-badge]][CI] [![PyPI Publish][PyPI-publish-badge]][PyPi-publish] [![codecov][codecov-badge]][codecov]        |
| Docs    | [![Read the Docs (version)][Docs-dev-badge]][Docs-dev] [![Read the Docs (version)][Docs-release-badge]][Docs-release] |
| Package | [![PyPI - Version][PyPI-badge]][PyPI] [![Conda][Conda-badge]][Conda] [![PyPI - Python Version][Python-badge]][PyPI]   |
| Meta    | [![DOI][DOI-badge]][DOI] [![License - BSD3][License-badge]][License] [![Pixi Badge][pixi-badge]][pixi-url]            |

[CI-badge]: https://img.shields.io/github/actions/workflow/status/munechika-koyo/cherab_lhd/ci.yaml?style=flat-square&logo=GitHub&label=CI
[CI]: https://github.com/munechika-koyo/cherab_lhd/actions/workflows/ci.yaml
[PyPI-publish-badge]: https://img.shields.io/github/actions/workflow/status/munechika-koyo/cherab_lhd/pypi-publish.yaml?style=flat-square&label=PyPI%20Publish&logo=github
[PyPI-publish]: https://github.com/munechika-koyo/cherab_lhd/actions/workflows/pypi-publish.yaml
[codecov]: https://codecov.io/github/munechika-koyo/cherab_lhd
[codecov-badge]: https://img.shields.io/codecov/c/github/munechika-koyo/cherab_lhd?token=05LZGWUUXA&style=flat-square&logo=codecov
[Docs-dev-badge]: https://img.shields.io/readthedocs/cherab-lhd/latest?style=flat-square&logo=readthedocs&label=dev%20docs
[Docs-dev]: https://cherab-lhd.readthedocs.io/en/latest/?badge=latest
[Docs-release-badge]: https://img.shields.io/readthedocs/cherab-lhd/stable?style=flat-square&logo=readthedocs&label=release%20docs
[Docs-release]: https://cherab-lhd.readthedocs.io/en/stable/?badge=stable
[PyPI-badge]: https://img.shields.io/pypi/v/cherab-lhd?label=PyPI&logo=pypi&logoColor=gold&style=flat-square
[PyPI]: https://pypi.org/project/cherab-lhd/
[Conda-badge]: https://img.shields.io/conda/vn/conda-forge/cherab-lhd?logo=conda-forge&style=flat-square
[Conda]: https://prefix.dev/channels/conda-forge/packages/cherab-lhd
[Python-badge]: https://img.shields.io/pypi/pyversions/cherab-lhd?logo=Python&logoColor=gold&style=flat-square
[DOI-badge]: https://zenodo.org/badge/DOI/10.5281/zenodo.14929182.svg
[DOI]: https://doi.org/10.5281/zenodo.14929182
[License-badge]: https://img.shields.io/github/license/munechika-koyo/cherab_lhd?style=flat-square
[License]: https://opensource.org/licenses/BSD-3-Clause
[pixi-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json&style=flat-square
[pixi-url]: https://pixi.sh

---

This repository contains the Large Helical Device (LHD) machine-dependent extensions of [`cherab`](https://www.cherab.info/) code.
LHD is a helical magnetic confinement fusion device in the National Institute for Fusion Science (NIFS) in Japan.

## 🌐 Installation

You can install the package from PyPI:

```bash
pip install cherab-lhd
```

Or from Conda:

```bash
# Using mamba
mamba install -c conda-forge cherab-lhd

# Using pixi
pixi add cherab-lhd
```

## 🔨 For Developers

### Pre-requisites

- [`pixi`](https://prefix.dev), a tool for project and package management.

### ⚡️ Task-based execution

We offer some tasks to execute programs in CLI.
You can see the list of tasks using [pixi](https://pixi.sh) command.

```shell
pixi task list
```

If you want to execute a task, you can use the following command.

```shell
pixi run <task_name>
```

### 🚀 Notebooks

We provide some notebooks to demonstrate the usage of the CHERAB-LHD code.
To launch the Jupyter lab server, you can use the following command.

```shell
pixi run lab
```

Then, you can access the Jupyter lab server from your web browser.

## 📝 Documentation

The [documentation](https://cherab-lhd.readthedocs.io/) is made with [Sphinx](https://www.sphinx-doc.org/en/master/) and hosted on [Read the Docs](https://readthedocs.org/).
There are two versions of the documentation:

- [Development](https://cherab-lhd.readthedocs.io/en/latest/)
- [Release](https://cherab-lhd.readthedocs.io/en/stable/)

You can build the documentation locally by running the following command.

```shell
# Build the documentation
pixi run doc-build

# Serve the documentation locally
pixi run doc-serve
```

The documentation is hosted locally at `http://localhost:8000/`.

## 📄 License

`cherab-lhd` is distributed under the terms of the [BSD-3-Clause license](https://opensource.org/licenses/BSD-3-Clause).
