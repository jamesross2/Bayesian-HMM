import setuptools

setuptools.setup(
    name='bayesian_hmm',
    version='0.0.0a0',
    license='MIT',
    packages=setuptools.find_packages(exclude=['tests']),
    author='James Ross',
    author_email='jamespatross@gmail.com',
    description='A non-parametric approach to Hidden Markov Models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/jamesross2/Bayesian-HMM',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'],
    setup_requires=['nose>=1.0'],
    test_suite='nose.collector')
