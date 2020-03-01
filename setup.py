import setuptools

setuptools.setup(
    name="bayesian_hmm",
    version="0.1.0",
    license="MIT",
    packages=setuptools.find_packages(exclude=["tests"]),
    author="James Ross",
    author_email="jamespatross@gmail.com",
    description="A non-parametric Bayesian approach to hidden Markov models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jamesross2/Bayesian-HMM",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
        'Programming Language :: Python :: 3.8'
    ],
    install_requires=["scipy", "numpy", "terminaltables", "tqdm", "sympy"],
    test_suite="py.test",
    tests_require=["pytest", "pytest-cov", "black", "codecov", "isort", "darglint"],
)
