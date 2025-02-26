# CHERAB-LHD

|         |                                                                                                                     |
| ------- | ------------------------------------------------------------------------------------------------------------------- |
| CI/CD   | [![pre-commit.ci status][pre-commit-ci-badge]][pre-commit-ci] [![PyPI Publish][PyPI-publish-badge]][PyPi-publish] [![codecov][codecov-badge]][codecov] |
| Docs    | [![Documentation Status][Docs-badge]][Docs]                                                                         |
| Package | [![PyPI - Version][PyPI-badge]][PyPI] [![Conda][Conda-badge]][Conda] [![PyPI - Python Version][Python-badge]][PyPI] |
| Meta    | [![DOI][DOI-badge]][DOI] [![License - BSD3][License-badge]][License] [![Pixi Badge][pixi-badge]][pixi-url]          |

[pre-commit-ci-badge]: https://results.pre-commit.ci/badge/github/munechika-koyo/cherab_lhd/main.svg
[pre-commit-ci]: https://results.pre-commit.ci/latest/github/munechika-koyo/cherab_lhd/main
[PyPI-publish-badge]: https://img.shields.io/github/actions/workflow/status/munechika-koyo/cherab_lhd/deploy-pypi.yml?style=flat-square&label=PyPI%20Publish&logo=github
[PyPI-publish]: https://github.com/munechika-koyo/cherab_lhd/actions/workflows/deploy-pypi.yml
[codecov]: https://codecov.io/github/munechika-koyo/cherab_lhd
[codecov-badge]: https://img.shields.io/codecov/c/github/munechika-koyo/cherab_lhd?token=05LZGWUUXA&style=flat-square&logo=codecov
[Docs-badge]: https://readthedocs.org/projects/cherab-lhd/badge/?version=latest&style=flat-square
[Docs]: https://cherab-lhd.readthedocs.io/en/latest/?badge=latest
[PyPI-badge]: https://img.shields.io/pypi/v/cherab-lhd?label=PyPI&logo=pypi&logoColor=gold&style=flat-square
[PyPI]: https://pypi.org/project/cherab-lhd/
[Conda-badge]: https://img.shields.io/conda/vn/conda-forge/cherab-lhd?logo=conda-forge&style=flat-square
[Conda]: https://prefix.dev/channels/conda-forge/packages/cherab-lhd
[Python-badge]: https://img.shields.io/pypi/pyversions/cherab-lhd?logo=Python&logoColor=gold&style=flat-square
[DOI-badge]: https://zenodo.org/badge/DOI/10.5281/zenodo.10118752.svg
[DOI]: https://doi.org/10.5281/zenodo.10118752
[License-badge]: https://img.shields.io/github/license/munechika-koyo/cherab_lhd?style=flat-square
[License]: https://opensource.org/licenses/BSD-3-Clause
[pixi-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json&style=flat-square
[pixi-url]: https://pixi.sh

---

This repository contains the Large Helical Device (LHD) machine-dependent extensions of [`cherab`](https://www.cherab.info/) code.
LHD is a helical magnetic confinement fusion device in the National Institute for Fusion Science (NIFS) in Japan.

## Table of Contents

- [Get Started](#get-started)
- [Documentation](#documentation)
- [License](#license)

## Get Started

### Task-based execution
We offer some tasks to execute programs in CLI.
You can see the list of tasks using [pixi](https://pixi.sh) command.

```console
pixi task list
```

If you want to execute a task, you can use the following command.

```console
pixi run <task_name>
```

### Notebooks
We provide some notebooks to demonstrate the usage of the CHERAB-NAGDIS code.
To launch the Jupyter lab server, you can use the following command.

```console
pixi run lab
```
Then, you can access the Jupyter lab server from your web browser.

## Documentation
<!-- The [documentation]() is made with GitHub Actions and hosted by [GitHub Pages](https://docs.github.com/pages). -->
The documentation is currently under preparation.

## License
`cherab-lhd` is distributed under the terms of the [BSD-3-Clause license](https://opensource.org/licenses/BSD-3-Clause).
