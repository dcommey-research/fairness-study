#!/bin/sh
set -eu
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
"$PYTHON_BIN" -m revision.fetch_data
"$PYTHON_BIN" -m pytest
"$PYTHON_BIN" main.py --output results/revision_verified
"$PYTHON_BIN" analysis.py --input results/revision_verified
"$PYTHON_BIN" -m revision.verify --input results/revision_verified
"$PYTHON_BIN" -m revision.extended.compas_encoding
"$PYTHON_BIN" -m revision.extended.publication
"$PYTHON_BIN" -m revision.extended.verify_compas
