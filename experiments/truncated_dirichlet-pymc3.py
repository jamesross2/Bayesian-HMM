#!/usr/bin/env python
"""Use PyMC3 to demonstrate a truncated Dirichlet Process implementation.

This script has leans heavily on this fantastic tutorial: https://www.pymc.io/projects/examples/en/latest/mixture_models/dp_mix.html
Refer there for any questions.

PyMC is a well-developed and managed package, and we strongly suggest that users consider using this. For most practical applications,
a truncated Dirichlet Procees BHMM is almost surely sufficient, and PyMC 4 contains numerous optimizations for sampling etc. that 
make it a very good choice.

This script could be made more efficient in PyMC4: you can use StickBreakingWeights directly, for example. However, it was not clear
how to resolve categorical priors; see here: https://discourse.pymc.io/t/using-a-categorical-prior-for-a-mixture-model/5394/6
"""

import pymc3 as pm
import theano.tensor as tt


def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining

# create some fake data (real data may need to be made equal length, or use a more sophisticated model specification)
chains = [
    [0, 1, 2, 3],
    [0, 1, 2, 3, 0, 1, 2, 2, 0, 1, 3, 0]
]

# Define meta-parameters of BHMM
K = 6  # number of latent states
N = len(set(s for chain in chains for s in chain))  # number of observed states

model = pm.Model(coords={"states": range(K), "observations": range(N), "obs_id": range(N)})

with model:
    alpha = pm.Gamma("alpha", 1.0, 1.0)
    gamma = pm.Gamma("gamma", 1.0, 1.0)
    
    # transition matrix: each row is a Dirichlet rv
    beta_transition = pm.Beta("beta_transition", 3.0, alpha, dims="states")
    w_transition = pm.Deterministic("w_transition", stick_breaking(beta_transition), dims="states")
    pi = [pm.Dirichlet(f"pi_{state}", gamma * w_transition, dims="states") for state in range(K)]
    # pi = tt.real(pi)
    
    # emission matrix: another Dirichlet distribution
    beta_emission = pm.Gamma("beta_emission", 1.0, 1.0)
    emissions = [pm.Dirichlet(f"emission_{state}", [beta_emission for _ in range(N)], dims="observations") for state in range(K)]
    # emissions = tt.real([tt.real(e) for e in emissions])
    
    # now, create latent state chain
    chain_states = [None for _ in chains]
    chain_emissions = [None for _ in chains]
    for i, chain in enumerate(chains):
        chain_states[i] = [pm.Categorical(f"s_{i}_0", p = [1 / N for _ in range(N)])]
        for t in range(1, len(chain)):
            chain_states[i].append(pm.Categorical(f"s_{i}_{t}", p=tt.real(pi)[chain_states[i][t-1]]))
        # now, tie observations to latent states
        chain_emissions[i] = [pm.Categorical(f"e_{i}_{t}", p=tt.real(emissions)[chain_states[i][t]], observed=chain[t]) for t in range(len(chain))]

    # time for fitting
    trace = pm.sample(500, tune=100)

