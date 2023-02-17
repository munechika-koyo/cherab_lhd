#!/bin/bash

echo Install package from source...

# install rich_click & tomli to allow dev.py CLI
python -m pip install rich_click tomli

# install build dependencies
python dev.py install-deps

# install package
python -m pip install --no-build-isolation .[dev,docs]

echo Building docs...

python dev.py doc
