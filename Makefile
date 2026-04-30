# Chess behavioral anomaly — local dev & CI helpers
# Requires: Python 3.10+ on PATH as `python3` (override with PYTHON=...)

PYTHON ?= python3
VENV  := .venv
PY    := $(VENV)/bin/python
PIP   := $(VENV)/bin/pip

.PHONY: help setup venv install download test lab pipeline clean

help:
	@echo "Targets:"
	@echo "  make setup     - venv + pip install + optional Kaggle download (idempotent)"
	@echo "  make install   - create .venv and install requirements only"
	@echo "  make download  - fetch data/raw/games.csv if Kaggle is configured"
	@echo "  make test      - pytest (uses .venv if present)"
	@echo "  make lab       - Jupyter Lab → 01_eda.ipynb"
	@echo "  make pipeline  - python -m src.pipeline"
	@echo "  make clean     - remove .venv (does not delete data/)"

venv:
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)

install: venv
	$(PY) -m pip install -U pip setuptools wheel
	$(PY) -m pip install -r requirements.txt

download: venv
	$(PY) scripts/download_kaggle_dataset.py

setup: install download
	@echo ""
	@echo "Activate:  source $(VENV)/bin/activate"
	@echo "Windows:   .venv\\Scripts\\activate"

test:
	@if [ -x $(PY) ]; then $(PY) -m pytest; else $(PYTHON) -m pytest; fi

lab:
	@if [ -x $(PY) ]; then $(PY) -m jupyter lab notebooks/01_eda.ipynb; else $(PYTHON) -m jupyter lab notebooks/01_eda.ipynb; fi

pipeline:
	@if [ -x $(PY) ]; then $(PY) -m src.pipeline; else $(PYTHON) -m src.pipeline; fi

clean:
	rm -rf $(VENV)
