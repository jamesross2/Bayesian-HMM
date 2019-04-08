# Bayesian Hidden Markov Models

[![Build Status](https://travis-ci.org/jamesross2/Bayesian-HMM.svg?branch=master)](https://travis-ci.org/jamesross2/Bayesian-HMM)

This code implements a non-parametric Bayesian Hidden Markov model,
sometimes referred to as a Hierarchical Dirichlet Process Hidden Markov
Model (HDP-HMM), or an Infinite Hidden Markov Model (iHMM). This package has capability
for a standard non-parametric Bayesian HMM, as well as a sticky HDPHMM 
(see references). Inference is performed via Markov chain Monte Carlo estimation,
including efficient beam sampling for the latent sequence resampling steps,
and multithreading when possible for parameter resampling.


## Installation

The current version is development only, and installation is only recommended for
people who are aware of the risks. It can be installed through PyPI:

```sh
pip install bayesian-hmm
```


## Hidden Markov Models

[Hidden Markov Models](https://en.wikipedia.org/wiki/Hidden_Markov_model) 
are powerful time series models, which use latent variables to explain observed emission sequences.
The result is a generative model for time series data, which is often tractable and can be easily understood.
The latent series is assumed to be a Markov chain, which requires a starting distribution and transition distribution, 
as well as an emission distribution to tie emissions to latent states.
Traditional parametric Hidden Markov Models use a fixed number of states for the latent series Markov chain.
Hierarchical Dirichlet Process Hidden Markov Models (including the one implemented by `bayesian_hmm` package) allow
for the number of latent states to vary as part of the fitting process. 
This is done by using a hierarchical Dirichlet prior on the latent state starting and transition distributions, 
and performing MCMC sampling on the latent states to estimate the model parameters.


## Usage

Basic usage allows us to supply a list of emission sequences, initialise the HDPHMM, and perform MCMC estimation.
The below example constructs some artificial observation series, and uses a brief MCMC estimation step to estimate the 
model parameters.
We use a moderately sized data to showcase the speed of the package: 50 sequences of length 200, with 500 MCMC steps.   

```python
import bayesian_hmm

# create emission sequences
base_sequence = list(range(5)) + list(range(5, 0, -1))
sequences = [base_sequence * 20 for _ in range(50)]

# initialise object with overestimate of true number of latent states
hmm = bayesian_hmm.HDPHMM(sequences, sticky=False)
hmm.initialise(k=20)

# estimate parameters, making use of multithreading functionality
results = hmm.mcmc(n=500, burn_in=100)

# print final probability estimates (expect 10 latent states)
hmm.print_probabilities()
```

The `bayesian_hmm` package can handle more advanced usage, including:
  * Multiple emission sequences,
  * Emission series of varying length,
  * Any categorical emission distribution,
  * Multithreaded MCMC estimation, and
  * Starting probability estimation, which share a dirichlet prior with the transition probabilities.


## Inference

This code uses an MCMC approach to parameter estimation. 
We use efficient Beam sampling on the latent sequences, as well as 
Metropolis Hastings sampling on each of the hyperparameters.
We approximate true resampling steps by using probability estimates
calculated on all states of interest, rather than the 
leaving probabilities unadjusted
for current variable resampling steps (rather than removing the current)
variable for the sampled estimate. 


## Outstanding issues and future work

We have the following set as a priority to improve in the future:

* Expand package to include standard non-Bayesian HMM functions, such as Baum Welch and Viterbi algorithm
* Allow for missing or `NULL` emissions which do not impact the model probability
* Include functionality to use maximum likelihood estimates for the hyperparameters 
(currently only Metropolis Hastings resampling is possible for hyperparameters)


## References

Van Gael, J., Saatci, Y., Teh, Y. W., & Ghahramani, Z. (2008, July). Beam sampling for the infinite hidden Markov model. In Proceedings of the 25th international conference on Machine learning (pp. 1088-1095). ACM.

Beal, M. J., Ghahramani, Z., & Rasmussen, C. E. (2002). The infinite hidden Markov model. In Advances in neural information processing systems (pp. 577-584).

Fox, E. B., Sudderth, E. B., Jordan, M. I., & Willsky, A. S. (2007). The sticky HDP-HMM: Bayesian nonparametric hidden Markov models with persistent states. Arxiv preprint.
