#!/bin/bash

echo Install dependencies...

# install dependencies from github
source requirements/github_pkgs.sh
# for doc
pip install -r requirements/docs.txt

# install package

source dev/install.sh

echo Building docs...

cd docs

make html
