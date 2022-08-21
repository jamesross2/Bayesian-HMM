#!/usr/bin/make -f
SHELL = /bin/sh
.DELETE_ON_ERROR:

# boilerplate variables, do not edit
MAKEFILE_PATH := $(abspath $(firstword $(MAKEFILE_LIST)))
MAKEFILE_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# required values, set to defaults here if not given in config.mk
PACKAGE_DIR ?= src
TEST_DIR ?= tests
STUB_DIR ?= stubs
LINTING_LINELENGTH ?= 120
PYTHON ?= python
CODECOV_TOKEN ?= ${CODECOV_TOKEN}

CURRENT_SIGN_SETTING := $(shell git config commit.gpgSign)

# use regex to go through this Makefile and print help lines
# help lines is any comment starting with double '#' (see next comment). Prints alphabetical order.
## help :		print this help.
.PHONY: help
help: Makefile
	@echo "\nBayesian Hidden Markov Models."
	@echo "\n       Generic commands"
	@sed -n 's/^## /                /p' $< | sort

## clean:		Remove python artifacts.
.PHONY: clean-pyc clean-build clean
clean-pyc:
	find . -regex '^./\($(PACKAGE_DIR)\|$(TEST_DIR)\)/.*\.py[co]' -delete
	find . -regex '^./\($(PACKAGE_DIR)\|$(TEST_DIR)\)/.*__pycache__' -delete

clean-build:
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive *.egg-info

clean: clean-pyc clean-build

isort-test: clean-pyc
	isort \
		--recursive \
		--check-only \
		--line-width $(LINTING_LINELENGTH) \
		--multi-line 3 \
		--trailing-comma \
		$(PACKAGE_DIR) $(TEST_DIR)

isort: clean-pyc
	isort \
		--recursive \
		--line-width $(LINTING_LINELENGTH) \
		--multi-line 3 \
		--trailing-comma \
		$(PACKAGE_DIR) $(TEST_DIR)

darglint-test:
	darglint --docstring-style google --strictness full $(PACKAGE_DIR) $(TEST_DIR)

black-test:
	black \
		--check \
	 	--include "^/($(PACKAGE_DIR)/|$(TEST_DIR)/).*\.pyi?" \
		--line-length $(LINTING_LINELENGTH) \
		.

black:
	black \
		--include "^/($(PACKAGE_DIR)/|$(TEST_DIR)/).*\.pyi?" \
		--line-length $(LINTING_LINELENGTH) \
		.
## stubs:		Run stub generation.
stubs: clean-pyc
	stubgen -o $(STUB_DIR) $(PACKAGE_DIR)

mypy-test:
	mypy $(PACKAGE_DIR)

pytest:
	python -m pytest \
		--verbose \
		--color=yes \
		--cov=$(PACKAGE_DIR) \
		--cov-report term-missing \
		--cov-report xml

## test:		Run all tests.
test: clean-pyc isort-test darglint-test black-test pytest

## format:		Apply all formatting tools.
format: clean-pyc isort black stubs

## profile:	Run cProfile on the MCMC example and print longest running functions.
profile:
	python -m cProfile -o .profile.txt "$(TEST_DIR)/example.py"
	python -c "import pstats; s = pstats.Stats('.profile.txt'); print(s.strip_dirs().sort_stats('cumtime').print_stats(20))"

## tox:		Run tox testing.
tox:
	tox

## release:	Build and upload to PyPI.
release: clean-pyc clean-build test
	git config commit.gpgSign true
	bumpversion $(bump)
	git push upstream && git push upstream --tags
	python setup.py sdist bdist_wheel
	twine upload dist/*
	git config commit.gpgSign "$(CURRENT_SIGN_SETTING)"

