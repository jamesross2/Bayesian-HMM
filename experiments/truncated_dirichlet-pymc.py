#!/usr/bin/env python
"""Use PyMC 4.0 to demonstrate a truncated Dirichlet Process implementation.

This script has leans heavily on this fantastic tutorial: https://www.pymc.io/projects/examples/en/latest/mixture_models/dp_mix.html
Refer there for any questions.

PyMC is a well-developed and managed package, and we strongly suggest that users consider using this. For most practical applications,
a truncated Dirichlet Procees BHMM is almost surely sufficient, and PyMC 4 contains numerous optimizations for sampling etc. that 
make it a very good choice.

This script could be made more efficient in PyMC4: you can use StickBreakingWeights directly, for example. However, it was not clear
how to resolve categorical priors; see here: https://discourse.pymc.io/t/using-a-categorical-prior-for-a-mixture-model/5394/6
"""

# weird import with numpy is required for theano tensor in pymc 4.0
import pymc as pm
import numpy
import numpy.distutils
numpy.distutils.__config__.blas_opt_info = numpy.distutils.__config__.blas_ilp64_opt_info
import theano.tensor as tt


# create some fake data with integers in range(N)
chains = [
    [0, 1, 2, 3],
    [0, 1, 2, 3, 0, 1, 2, 2, 0, 1, 3, 0]
]

# Define meta-parameters of BHMM
K = 8  # number of latent states
N = len(set(s for chain in chains for s in chain))  # number of observed states

with pm.Model(coords={"states": range(K), "observations": range(N), "obs_id": range(N)}) as model:
    alpha = pm.Gamma("alpha", 2.0, 2.0)
    gamma = pm.Gamma("gamma", 3.0, 3.0)
    beta_emission = pm.Gamma("beta_emission", 1.0, 1.0)
    
    # transition matrix: each row is a Dirichlet rv
    w_transition = pm.StickBreakingWeights("beta_transition", alpha=alpha, K=K, dims="states")  # only in PyMC 4
    pi = [pm.Dirichlet(f"pi_{state}", gamma * w_transition, dims="states") for state in range(K)]
    
    # emission matrix: another Dirichlet distribution
    emissions = [pm.Dirichlet(f"emission_{state}", [beta_emission for _ in range(N)], dims="observations") for state in range(K)]
    
    # now, create latent state chain
    # states = [pm.Categorical("s0", p = [1 / N for _ in range(N)])]
    # for t in range(1, T):
    #     # get transitions from expanding out, since PyMC cannot index by random variables
    #     # prob = pm.Deterministic(f"prob_{t}", [row * state for row, state in zip(pi, states[t-1])])
    #     # states.append(pm.Categorical(f"s{t}", p=prob))
    #     # states.append(pm.Categorical(f"s{t}", p=pi[states[t-1]]))
    #     states.append(pm.Categorical(f"s{t}", p=pi[states[t-1].eval()]))
    
    # now, tie observations to latent states
    # emissions = [pm.Categorical(f"e{t}", p=emissions[states[t].eval()], observed=chain) for t in range(T)]

    # now, create latent state chain
    chain_states = [None for _ in chains]
    chain_emissions = [None for _ in chains]
    for i, chain in enumerate(chains):
        chain_states[i] = [pm.Categorical(f"s_{i}_0", p = [1 / N for _ in range(N)])]
        for t in range(1, len(chain)):
            chain_states[i].append(pm.Categorical(f"s_{i}_{t}", p=pi[chain_states[i][t-1]]))
        
        # now, tie observations to latent states
        chain_emissions[i] = [pm.Categorical(f"e_{i}_{t}", p=emissions[chain_states[i][t]], observed=chain[t]) for t in range(len(chain))]

    # time for fitting
    trace = pm.sample(500, tune=100, cores=1)
