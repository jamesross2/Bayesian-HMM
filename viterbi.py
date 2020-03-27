#!/usr/bin/env python3
"""Implement the Viterbi algorithm to calculate most likely sequence of hidden states."""

import numpy as np


def viterbi(observations, states, start_p, trans_p, emit_p) -> np.ndarray:
    """Implement the Viterbi algorithm as outlined on [Wikipedia](https://en.wikipedia.org/wiki/Viterbi_algorithm) for now. Later, generalize as needed."""

    # TODO: Think about making states and observations numerical for easier implementation below.
    # I.e., removing the `enumerate()`s.

    # Pre-allocate matrices for T1, T2.
    T1 = np.empty((len(states), len(observations)), dtype=np.float)
    T2 = np.empty((len(states), len(observations)), dtype=np.float)

    # Initialize states.
    for i, state in enumerate(states):
        T1[i, 0] = start_p[state] * emit_p[state][observations[i]]
        T2[i, 0] = 0.0

    # Intermediate states.
    for j, obs in enumerate(observations[1:]):
        for i, state in enumerate(states):

            # TODO: think about using a tee and a generator here for long/memory-intensive sequences.
            candidates = [
                T1[k, j - 1] * trans_p[_state][state] * emit_p[state][obs]
                for k, _state in enumerate(states)
            ]
            T1[i, j] = np.max(candidates)
            T2[i, j] = np.argmax(candidates)

    # Step back through the calculations and find the most likely sequence.
    Z = np.empty((len(observations),), dtype=np.float)
    X = np.empty((len(observations),), dtype=str)

    Z[-1] = np.argmax(T1[:, -1])
    X[-1] = states[int(Z[-1])]
    for j in range(len(observations) - 1, 0, -1):  # FIX: ugly use of `range()`.
        Z[j - 1] = T2[int(Z[j]), j]
        X[j - 1] = states[int(Z[j - 1])]

    return X


if __name__ == "__main__":
    observations = ["normal", "cold", "dizzy"]
    states = ["Healthy", "Fever"]

    start_p = {"Healthy": 0.6, "Fever": 0.4}
    trans_p = {
        "Healthy": {"Healthy": 0.7, "Fever": 0.3},
        "Fever": {"Healthy": 0.4, "Fever": 0.6},
    }
    emit_p = {
        "Healthy": {"normal": 0.5, "cold": 0.4, "dizzy": 0.1},
        "Fever": {"normal": 0.1, "cold": 0.3, "dizzy": 0.6},
    }

    result = viterbi(observations, states, start_p, trans_p, emit_p)
    print(result)
