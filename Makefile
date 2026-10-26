UV := uv
PYTHON := python
CUDA := CUDA_VISIBLE_DEVICES=""

.PHONY: install run cli debug clean lint lint-strict test

install:
	$(UV) sync

run:
	$(CUDA) $(UV) run $(PYTHON) -m src

cli:
	$(CUDA) $(UV) run $(PYTHON) -m src --visualize

debug:
	$(UV) run $(PYTHON) -m pdb -c continue -m src

clean:
	rm -rf .mypy_cache .pytest_cache .cache data/output \
		src/__pycache__ tests/__pycache__ llm_sdk/llm_sdk/__pycache__ \
		call_me_maybe.egg-info

lint:
	$(UV) run flake8 src
	$(UV) run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs 

lint-strict:
	$(UV) run flake8 src
	$(UV) run mypy src --strict

test:
	$(UV) run pytest -v
