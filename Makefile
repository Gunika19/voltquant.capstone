.PHONY: env-data env-classical env-quantum test lint clean all

UV := uv
PYTHON_VERSION := 3.11

# .venv-data is the environment all pipeline scripts (s01..s06) and tests
# run against -- see envs/data.txt. classical/quantum are separate, mutually
# exclusive venvs (see envs/classical.txt, envs/quantum.txt for why).

env-data:
	$(UV) venv .venv-data --python $(PYTHON_VERSION)
	$(UV) pip install --python .venv-data/bin/python -r envs/data.txt

env-classical:
	$(UV) venv .venv-classical --python $(PYTHON_VERSION)
	$(UV) pip install --python .venv-classical/bin/python -r envs/classical.txt

env-quantum:
	$(UV) venv .venv-quantum --python $(PYTHON_VERSION)
	$(UV) pip install --python .venv-quantum/bin/python -r envs/quantum.txt

test:
	.venv-data/bin/python -m pytest packages -q

lint:
	$(UV) tool run ruff check packages

clean:
	rm -rf .venv-data .venv-classical .venv-quantum
	find . -name "__pycache__" -not -path "./.git/*" -exec rm -rf {} +
	find . -name "*.egg-info" -not -path "./.git/*" -exec rm -rf {} +
	find . -name ".pytest_cache" -not -path "./.git/*" -exec rm -rf {} +
	rm -rf reports/figures/*

all: env-data test lint
