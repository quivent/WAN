PYTHON ?= python3
UV ?= uv
VENV ?= .venv
PROMPT ?= a slow cinematic push through a rainy neon market
TASK ?= t2v-A14B
SIZE ?= 1280x720
GPUS ?= 8

VENV_PY := $(VENV)/bin/python

.PHONY: help setup doctor plan enqueue worker jobs check

help:
	@echo "Targets:"
	@echo "  make setup    Create .venv and install dependencies"
	@echo "  make doctor   Inspect CUDA, torch, and WAN paths"
	@echo "  make plan     Print a native Wan2.2 command"
	@echo "  make enqueue  Add one job to the local queue"
	@echo "  make worker   Run the continuous queue worker"
	@echo "  make jobs     Show queue state"
	@echo "  make check    Compile Python modules"

setup:
	test -d $(VENV) || $(UV) venv $(VENV) --python $(PYTHON)
	$(UV) pip install --python $(VENV_PY) -r requirements.txt
	$(VENV_PY) -m pip install -e .

doctor:
	$(VENV_PY) -m wanctl.cli doctor

plan:
	$(VENV_PY) -m wanctl.cli plan "$(PROMPT)" --task $(TASK) --size $(SIZE) --gpus $(GPUS)

enqueue:
	$(VENV_PY) -m wanctl.cli enqueue "$(PROMPT)" --task $(TASK) --size $(SIZE) --gpus $(GPUS)

worker:
	$(VENV_PY) -m wanctl.cli worker

jobs:
	$(VENV_PY) -m wanctl.cli jobs --verbose

check:
	$(PYTHON) -m compileall -q wanctl
