UV := uv
PYTHON := python

.PHONY: install run debug clean lint lint-strict

install:
	$(UV) sync

run:
	$(UV) run $(PYTHON) -m src

debug:
	$(UV) run $(PYTHON) -m pdb -c continue -m src

clean:
	rm -rf .mypy_cache .pytest_cache src/__pycache__ llm_sdk/llm_sdk/__pycache__ call_me_maybe.egg-info data/output

lint:
	$(UV) run flake8 .
	$(UV) run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run flake8 .
	$(UV) run mypy . --strict
