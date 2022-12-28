# CHERAB LHD

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/munechika-koyo/cherab_lhd/main.svg?badge_token=Lmet0dXxT2iOgJ_pHNunBw)](https://results.pre-commit.ci/latest/github/munechika-koyo/cherab_lhd/main?badge_token=Lmet0dXxT2iOgJ_pHNunBw)
[![Netlify Status](https://api.netlify.com/api/v1/badges/af76f666-95b7-4282-85f7-42dacd2d97f2/deploy-status)](https://app.netlify.com/sites/cherab-lhd/deploys)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Docstring formatter: docformatter](https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg)](https://github.com/PyCQA/docformatter)
[![Docstring style: numpy](https://img.shields.io/badge/%20style-numpy-459db9.svg)](https://numpydoc.readthedocs.io/en/latest/format.html)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)


CHERAB for Large Herical Device in National Institute for Fusion Science

Please see our [documentation](https://cherab-lhd.netlify.app/)
for guidance on using the code.

Quick installation
-------------------
Synchronize `cherab-lhd` source:

```Shell
git clone https://github.com/munechika-koyo/cherab_lhd.git
```
And, download Large data files by running the following command at the source root:
```Shell
git lfs install
git lfs fetch
```
Then, install `cherab-lhd` with pip:
```Shell
pip install .
```

For developper
---
Create a conda development environment, build `cherab-lhd` with `Meson`
```Shell
conda env create -f environment.yaml
conda activate cherab-lhd-dev
python dev.py build
python dev.py install
```
