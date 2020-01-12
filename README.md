# Bayesian Hidden Markov Models

[![Build Status](https://img.shields.io/travis/jamesross2/Bayesian-HMM/master?logo=travis&style=flat-square)](https://travis-ci.org/jamesross2/Bayesian-HMM?style=flat-square)
[![PyPI Version](https://img.shields.io/pypi/v/bayesian-hmm?label=PyPI&logo=pypi&style=flat-square)](https://pypi.org/project/bayesian-hmm/)
[![Code Coverage](https://img.shields.io/codecov/c/github/jamesross2/Bayesian-HMM/master?logo=codecov&style=flat-square&label=codecov)](https://codecov.io/gh/jamesross2/Bayesian-HMM)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?logo=black&style=flat-square)](https://github.com/psf/black)

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

results = hmm.mcmc(n=500, burn_in=100, ncores=3, save_every=10, verbose=True)
```

This model typically converges to 10 latent states, a sensible posterior. In some cases,
it converges to 11 latent states, in which a starting state which outputs '0' with high
confidence is separate to another latent start which outputs '0' with high confidence.
We can inspect this using the printed output, or with probability matrices printed 
directly.

```python
# print final probability estimates (expect 10 latent states)
emissions, transitions = hmm.print_probabilities()
print(emissions)
```

We get something like the following:

```
╔Emission probabilities═══════╦════════╦════════╦════════╦════════╗
║ S_i \ E_i ║ 0      ║ 1      ║ 2      ║ 3      ║ 4      ║ 5      ║
╠═══════════╬════════╬════════╬════════╬════════╬════════╬════════╣
║         d ║ 0.0    ║ 0.0    ║ 1.0    ║ 0.0    ║ 0.0    ║ 0.0    ║
║         j ║ 0.9994 ║ 0.0    ║ 0.0    ║ 0.0006 ║ 0.0    ║ 0.0    ║
║         m ║ 0.0    ║ 0.0001 ║ 0.0009 ║ 0.9986 ║ 0.0001 ║ 0.0003 ║
║         g ║ 0.0013 ║ 0.9986 ║ 0.0    ║ 0.0    ║ 0.0001 ║ 0.0    ║
║         a ║ 0.0    ║ 0.9999 ║ 0.0    ║ 0.0    ║ 0.0    ║ 0.0    ║
║         s ║ 0.0    ║ 0.0    ║ 0.0    ║ 0.0    ║ 1.0    ║ 0.0    ║
║         h ║ 0.0009 ║ 0.0    ║ 0.0001 ║ 0.0    ║ 0.999  ║ 0.0    ║
║         o ║ 0.0002 ║ 0.0004 ║ 0.9967 ║ 0.0026 ║ 0.0001 ║ 0.0    ║
║         f ║ 0.0    ║ 0.0    ║ 0.0002 ║ 0.9998 ║ 0.0001 ║ 0.0    ║
║         i ║ 0.0    ║ 0.0032 ║ 0.0001 ║ 0.0    ║ 0.0    ║ 0.9967 ║
╚═══════════╩════════╩════════╩════════╩════════╩════════╩════════╝
```

This final command prints the transition and emission probabilities of the model after
MCMC using the [`terminaltables`](https://pypi.org/project/terminaltables/) package. The 
code below visualises the results using [`pandas`](https://pypi.org/project/pandas/)
and [`seaborn`](https://pypi.org/project/seaborn/). For simplicity, we will stick with
the returned MAP estimate, but a more complete analysis might use a more sophisticated
approach.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# plot the number of states as a histogram
sns.countplot(results['state_count'])
plt.title('Number of latent states')
plt.xlabel('Number of latent states')
plt.ylabel('Number of iterations')
plt.show()
```

![State counts](https://raw.githubusercontent.com/jamesross2/Bayesian-HMM/master/outputs/plot_state_count.png)

Of course, this is exactly what we expect--we set up the model to have 10 latent states (possibly 11, if the choice
of hyperparameters encourages a unique starting state.) For something more interesting, we can check which states
commonly appear first in the latent series.

```python
# plot the starting probabilities of the sampled MAP estimate
map_index = results['chain_loglikelihood'].index(max(results['chain_loglikelihood']))
transition_map = results["transition_probabilities"][0][bayesian_hmm.StartingState()]
states = [state for state in transition_map if not state.special]
sns.barplot(
    x=[str(state) for state in states],
    y=[transition_map[state] for state in states]
)
plt.title('Starting state probabilities')
plt.xlabel('Latent state')
plt.ylabel('MAP estimate for initial probability')
plt.show()
```

![MAP initial probabilities](https://raw.githubusercontent.com/jamesross2/Bayesian-HMM/master/outputs/plot_p_initial.png)

```python
# convert list of hyperparameters into a DataFrame
hyperparam_posterior_df = (
    pd.DataFrame(results['hyperparameters'])
    .rename({0: "beta_emission", 1: "alpha", 2: "gamma", 3: "kappa"}, axis=1)
    .reset_index()
    .melt(id_vars=['index'])
    .rename(columns={'index': 'iteration'})
)

# didn't make a sticky chain? Drop kappa
hyperparam_posterior_df.dropna(inplace=True)

hyperparam_prior_df = pd.concat(
    pd.DataFrame(
        {'iteration': range(500), 'variable': k, 'value': [v() for _ in range(500)]}
    )
    for k, v in hmm.priors.items()
    if v is not None
)
hyperparam_df = pd.concat(
    (hyperparam_prior_df, hyperparam_posterior_df), 
    keys=['prior', 'posterior'], 
    names=('type','index')
)
hyperparam_df.reset_index(inplace=True)

# advanced: plot sampled prior & sampled posterior together
g = sns.FacetGrid(
    hyperparam_df,
    col='variable', 
    col_wrap=3, 
    sharex=False,
    hue='type'
)
g.map(sns.distplot, 'value')
g.add_legend()
g.fig.suptitle('Hyperparameter prior & posterior estimates')
plt.subplots_adjust(top=0.9)
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()
```


![Hyperparameter posteriors](https://raw.githubusercontent.com/jamesross2/Bayesian-HMM/master/outputs/plot_hyperparameters.png)

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
