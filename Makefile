CURRENT_SIGN_SETTING := $(shell git config commit.gpgSign)
LINTING_IGNORED="venv/|__pycache__/|build/|_build|dist/|\.eggs|\.tox"
DARGLINT_INCLUDED=bayesian_hmm
CLEANING_DIR_REGEX=\(bayesian_hmm\|tests\)
BLACK_LINELENGTH=120

.PHONY: help clean-pyc clean-build isort lint lint-docstring coverage test release

help:
	@echo "    help: Print this help"
	@echo "    clean-pyc: Remove python artifacts."
	@echo "    clean-build: Remove build artifacts."
	@echo "    isort: Sort import statements."
	@echo "    lint: Apply black formatting to project."
	@echo "    lint-docstring: Check whether comments meet formatting requirements."
	@echo "    coverage: Run code coverage test."
	@echo "    test: Run pytest suite."
	@echo "    build: TODO: write make statement here."

clean-pyc:
	find . -regex '^./$(CLEANING_DIR_REGEX).*\.py[co]' -delete
	find . -wholename './__pycache__' -delete
	find . -regex '^./$(CLEANING_DIR_REGEX).*__pycache__' -delete
	find . -name '*~' -delete

clean-build:
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive *.egg-info

isort:
	isort --recursive bayesian_hmm
	isort --recursive tests

lint: isort
	black \
	 	--exclude=$(LINTING_IGNORED) \
		--line-length $(BLACK_LINELENGTH) \
		.

lint-docstring:
	darglint --docstring-style google --strictness full $(DARGLINT_INCLUDED)

coverage: clean-pyc
	python3 -m pytest \
		--verbose \
		--color=yes \
		--cov=bayesian_hmm/ \
		--cov-report term-missing

test: clean-pyc
	# check linting
	black \
		--exclude=$(LINTING_IGNORED) \
		--line-length $(BLACK_LINELENGTH) \
		--check \
		.

	# check comment style
	darglint --docstring-style google --strictness short $(DARGLINT_INCLUDED)
	
	# run tests with code coverage
	python3 -m pytest \
		--color=yes \
		--cov=bayesian_hmm

release: clean-pyc clean-build test
	git config commit.gpgSign true
	bumpversion $(bump)
	git push upstream && git push upstream --tags
	python setup.py sdist bdist_wheel
	twine upload dist/*
	git config commit.gpgSign "$(CURRENT_SIGN_SETTING)"

