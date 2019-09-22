#!/usr/bin/env python3
"""
Basic script to test typehinting in Bayesian-HMM package
"""
import bayesian_hmm

# create emission sequences
base_sequence = list(range(5)) + list(range(5, 0, -1))
sequences = [base_sequence * 20 for _ in range(50)]

# initialise object with overestimate of true number of latent states
hmm = bayesian_hmm.HDPHMM(sequences, sticky=False)
hmm.initialise(k=20)

# use MCMC to estimate the posterior
results = hmm.mcmc(n=500, burn_in=100, ncores=3, save_every=10, verbose=True)

map_index = results['chain_loglikelihood'].index(min(results['chain_loglikelihood']))
state_count_map = results['state_count'][map_index]
hyperparameters_map = results['hyperparameters'][map_index]
parameters_map = results['parameters'][map_index]
