# Bayesian Hidden Markov Models

This code implements a non-parametric Bayesian Hidden Markov model,
sometimes referred to as a Hierarchical Dirichlet Process Hidden Markov
Model, or an Infinite Hidden Markov Model. This package has capability
for a standard non-parametric Bayesian HMM, as well as a sticky HDPHMM 
(see references). Inference is performed via Markov chain Monte Carlo estimation,
including efficient beam sampling for the latent sequences resampling steps.
Currently, only categorical emission distributions are supported.
Resampling is also multithreaded for better performance.


## Installation

The current version is development only, and installation is only recommended for
people who are aware of the risks. It can be installed using the following code:

```
sudo apt-get install git
pip install git+https://github.com/jamesross2/Bayesian-HMM
```


## Hidden Markov Models

[Hidden Markov Models](https://en.wikipedia.org/wiki/Hidden_Markov_model) 
implement an efficient and tractable latent time series model.
They are used to explain observed sequences of data using a series of 'latent states'.
The latent series are explained via a starting distribution, transition distribution, and emission distribution.
In basic settings, the number of latent states is fixed at $K$.
This package uses a non-parametric Bayesian approach to identify the latent states,
using a hierarchical Dirichlet prior, where the number of latent states is explained
by a Dirichlet process.


## Usage

The class implements a Hierarchical Dirichlet Process Hidden Markov Model.
The standard way to estimate the parameters, via MCMC, is outlined below.
  

```
import bayesian_hmm

# create emission sequences
sequences = list(range(6)) + list(range(5,-1))
sequences = [sequences * 10] * 10

# initialise object with overestimate of true number of latent states
hmm = bayesian_hmm.HDPHMM(sequences)
hmm.initialise(k=20)

# estimate parameters
results = hmm.mcmc(n=100, burn_in=20)

# print final probability estimates
hmm.print_probabilities()
```

The package can handle slightly more advanced usage. 
It has support for the following:

  * Categorical emission distributions
  * Multiple emission sequences of different lengths, estimated in parallel
  * Starting probabilities, which share a dirichlet prior with the transition probabilities

For examples of each, consult the example scripts.


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

* Check all resampling processes
* Expand package to include standard non-Bayesian HMM functions, such as Baum Welch and Viterbi algorithm


## References

Van Gael, J., Saatci, Y., Teh, Y. W., & Ghahramani, Z. (2008, July). Beam sampling for the infinite hidden Markov model. In Proceedings of the 25th international conference on Machine learning (pp. 1088-1095). ACM.

Beal, M. J., Ghahramani, Z., & Rasmussen, C. E. (2002). The infinite hidden Markov model. In Advances in neural information processing systems (pp. 577-584).

Fox, E. B., Sudderth, E. B., Jordan, M. I., & Willsky, A. S. (2007). The sticky HDP-HMM: Bayesian nonparametric hidden Markov models with persistent states. Arxiv preprint.
