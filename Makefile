.PHONY: test install all

PYTHON := python3

install:
	pip install -r requirements.txt

test:
	$(PYTHON) tests/run_all_tests.py

all: install test
