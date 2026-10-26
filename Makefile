UV := uv
PYTHON := python

.PHONY: install run cli cache debug clean lint lint-strict

install:
	$(UV) sync

run:
	$(UV) run $(PYTHON) -m src

cli:
	$(UV) run $(PYTHON) -m src --visualize

cache:
	$(UV) run $(PYTHON) -m src --cache

debug:
	$(UV) run $(PYTHON) -m pdb -c continue -m src

clean:
	rm -rf .mypy_cache .cache data/output \
		src/__pycache__ llm_sdk/llm_sdk/__pycache__ \
		call_me_maybe.egg-info

lint:
	$(UV) run flake8 src
	$(UV) run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs 

lint-strict:
	$(UV) run flake8 src
	$(UV) run mypy src --strict
