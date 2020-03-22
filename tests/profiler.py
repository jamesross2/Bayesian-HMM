#!/usr/bin/env python3
"""Profile the MCMC iteration process.

We use this script to identify opportunities for optimisation within the package.
"""
import argparse
import cProfile
import pstats

import example


# import run_mcmc to run chain
def profile_mcmc(n_stats: int = 50, sort_key: str = "cumtime", **kwargs):
    # use basic settings for profiling
    arguments = {"emissions": 5, "repetitions": 10, "chains": 50, "states": 20, "iterations": 100, "ncores": 1}
    if any(key not in arguments.keys() for key in kwargs.keys()):
        raise ValueError("Unrecognised arguments in {}".format(kwargs.keys()))
    arguments.update(kwargs)

    profiler = cProfile.Profile()
    profiler.enable()
    example.mcmc(**arguments)
    profiler.disable()

    # run example with profiling enabled
    # with cProfile as profiler:
    #     example.mcmc(**arguments)

    # extract statistics from profiler
    stats = pstats.Stats(profiler).strip_dirs().sort_stats(sort_key)
    stats.print_stats(n_stats)
    return stats


if __name__ == "__main__":
    # get user arguments
    parser = argparse.ArgumentParser("Profile the Bayesian-HMM package examples.")
    parser.add_argument("--n_stats", help="Number of results to print [default = (%s)]", default=50)
    parser.add_argument("--sort_key", help="Statistic to sort results by [default = (%s)]", default="cumtime")
    arguments = vars(parser.parse_args())

    # run specified function
    profile_mcmc(**arguments)
