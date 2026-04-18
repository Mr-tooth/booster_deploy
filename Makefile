PYTHON ?= python
SRC_DIRS := booster_deploy tasks scripts tests

.PHONY: help ruff lint format test compile check

help:
	@echo "Available targets:"
	@echo "  make ruff    - run ruff checks"
	@echo "  make lint    - run lint + bytecode compile checks"
	@echo "  make format  - run ruff formatter"
	@echo "  make test    - run unit tests"
	@echo "  make check   - run lint and test"

ruff:
	$(PYTHON) -m ruff check $(SRC_DIRS)

lint: ruff compile

format:
	$(PYTHON) -m ruff format $(SRC_DIRS)

compile:
	$(PYTHON) -m compileall -q booster_deploy tasks scripts tests

test:
	PYTHONPATH=. $(PYTHON) -m unittest discover -s tests/controllers -p "test_*.py" -v

check: lint test
