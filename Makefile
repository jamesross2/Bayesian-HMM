CURRENT_SIGN_SETTING := $(shell git config commit.gpgSign)
PACKAGE_DIRECTORY=bayesian_hmm
TEST_DIRECTORY=tests
LINTING_LINELENGTH=120
STUB_DIRECTORY=stubs

.PHONY: help clean-pyc clean-build isort-test isort darglint-test black-test black stubs mypy-test pytest test format tox

help:
	@echo "    help: Print this help"
	@echo "    clean-pyc: Remove python artifacts."
	@echo "    clean-build: Remove build artifacts."
	@echo "    isort-test: Test whether import statements are sorted."
	@echo "    isort: Sort import statements."
	@echo "    darglint-test: Test whether docstrings are valid."
	@echo "    black-test: Test whether black formatting is adhered to."
	@echo "    black: Apply black formatting."
	@echo "    stubs: Run stub generation."
	@echo "    mypy-test: Test whether mypy type annotations are sufficient."
	@echo "    pytest: Run pytest suite."
	@echo "    test: Run all tests."
	@echo "    format: Apply all formatting tools."
	@echo "    tox: Run tox testing."
	@echo "    release: Build and upload to PyPI."

clean-pyc:
	find . -regex '^./\($(PACKAGE_DIRECTORY)\|$(TEST_DIRECTORY)\)/.*\.py[co]' -delete
	find . -regex '^./\($(PACKAGE_DIRECTORY)\|$(TEST_DIRECTORY)\)/.*__pycache__' -delete

clean-build:
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive *.egg-info

isort-test: clean-pyc
	isort \
		--recursive \
		--check-only \
		--line-width $(LINTING_LINELENGTH) \
		--multi-line 3 \
		--trailing-comma \
		$(PACKAGE_DIRECTORY) $(TEST_DIRECTORY)

isort: clean-pyc
	isort \
		--recursive \
		--line-width $(LINTING_LINELENGTH) \
		--multi-line 3 \
		--trailing-comma \
		$(PACKAGE_DIRECTORY) $(TEST_DIRECTORY)

darglint-test:
	darglint --docstring-style google --strictness full $(PACKAGE_DIRECTORY) $(TEST_DIRECTORY)

black-test:
	black \
		--check \
	 	--include "^/($(PACKAGE_DIRECTORY)/|$(TEST_DIRECTORY)/).*\.pyi?" \
		--line-length $(LINTING_LINELENGTH) \
		.

black:
	black \
		--include "^/($(PACKAGE_DIRECTORY)/|$(TEST_DIRECTORY)/).*\.pyi?" \
		--line-length $(LINTING_LINELENGTH) \
		.

stubs: clean-pyc
	stubgen -o $(STUB_DIRECTORY) $(PACKAGE_DIRECTORY)

mypy-test:
	mypy $(PACKAGE_DIRECTORY)

pytest:
	python -m pytest \
		--verbose \
		--color=yes \
		--cov=$(PACKAGE_DIRECTORY) \
		--cov-report term-missing

test: clean-pyc isort-test darglint-test black-test mypy-test pytest

format: clean-pyc isort black stubs

tox:
	tox

release: clean-pyc clean-build test
	git config commit.gpgSign true
	bumpversion $(bump)
	git push upstream && git push upstream --tags
	python setup.py sdist bdist_wheel
	twine upload dist/*
	git config commit.gpgSign "$(CURRENT_SIGN_SETTING)"

