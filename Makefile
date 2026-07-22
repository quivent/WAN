SHELL := /bin/zsh

PYTHON ?= python3.13
UV ?= uv
VENV ?= .venv
PROMPT ?= a slow cinematic push through a rainy neon market
TASK ?= t2v-A14B
SIZE ?= 1280x720
GPUS ?= 8

VENV_PY := $(VENV)/bin/python

.PHONY: help setup doctor plan check

help:
	@echo "Targets:"
	@echo "  make setup    Create .venv and install dependencies"
	@echo "  make doctor   Inspect CUDA, torch, and WAN paths"
	@echo "  make plan     Print a native Wan2.2 command"
	@echo "  make check    Compile Python modules"

setup:
	$(UV) venv $(VENV) --python $(PYTHON)
	$(UV) pip install --python $(VENV_PY) -r requirements.txt
	$(VENV_PY) -m pip install -e .

doctor:
	$(VENV_PY) -m wan.cli doctor

plan:
	$(VENV_PY) -m wan.cli plan "$(PROMPT)" --task $(TASK) --size $(SIZE) --gpus $(GPUS)

check:
	$(PYTHON) -m compileall -q wan
