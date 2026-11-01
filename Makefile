# Reproduction of the paper. Per-run data: RDT_DATA=/path/to/rdt_per_run_data (README.md, "Data").
PYTHON ?= python
export PYTHON

.PHONY: all install demo figures check clean

all: figures check

install:
	$(PYTHON) -m pip install -r requirements.txt

demo:
	$(PYTHON) run/run_pipeline.py --config run/configs/demo/demo_cifar100_4superclasses.json

figures:
	bash reproduce/figures.sh

check:
	$(PYTHON) reproduce/check.py

clean:
	rm -rf reproduce/out output
