#!/usr/bin/env python3
"""Run several standard MCMC iterations.

This is written to facilitate typehinting, profiling, and so on. It is a prototypical example
of a common use case.
"""
import argparse

import bayesian_hmm


def mcmc(emissions: int, repetitions: int, chains: int, states: int, iterations: int, **kwargs):
    # create emission sequences
    base_labels = list(range(emissions)) + list(range(emissions, 0, -1))
    base_sequence = [bayesian_hmm.State(label) for label in base_labels]
    sequences = [base_sequence * repetitions for _ in range(chains)]

    # initialise object with overestimate of true number of latent states
    hmm = bayesian_hmm.HDPHMM(sequences, sticky=False)
    hmm.initialise(k=states)

    # use MCMC to estimate the posterior
    results = hmm.mcmc(n=iterations, **kwargs)

    # finished! return results (never sure who wants them)
    return results


if __name__ == "__main__":
    # get user arguments
    parser = argparse.ArgumentParser("Run a prototypical MCMC process for a basic example.")
    parser.add_argument("--emissions", help="Number of emissions per subsequence [default = (%s)]", default=5)
    parser.add_argument("--repetitions", help="Times each subsequences is repeated [default = (%s)]", default=10)
    parser.add_argument("--chains", help="Number of chains [default = (%s)]", default=50)
    parser.add_argument("--states", help="Number of initial states[default = (%s)]", default=20)
    parser.add_argument("--iterations", help="Number of MCMC iterations [default = (%s)]", default=100)
    arguments = vars(parser.parse_args())

    # run specified function
    mcmc(**arguments)
