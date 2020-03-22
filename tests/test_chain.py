import bayesian_hmm
import bayesian_hmm.chain


def test_empty_chain():
    chain_empty = bayesian_hmm.Chain([])
    assert len(chain_empty) == 0


def test_print():
    # checks that printing does not cause an error
    emissions = [bayesian_hmm.State("e" + str(x)) for x in range(1000)]
    chain = bayesian_hmm.Chain(emissions)
    print(chain)
    repr(chain)
    assert True


def test_initialise_chain():
    emissions = [bayesian_hmm.State(val) for val in ("e0", "e1", "e2", "e3")]
    states = {bayesian_hmm.State(val) for val in ("s0", "s1", "s2", "s3", "s4")}  # intentionally different length
    chain = bayesian_hmm.Chain(emissions)

    # check chain initialised correctly
    assert chain.emission_sequence == emissions
    assert chain.T == len(chain)
    assert len(chain) == len(emissions)
    assert not chain.initialised_flag

    # initialise & check latent states in chain
    chain.initialise(states)
    assert len(chain) == len(emissions)
    assert chain.initialised_flag
    assert all(s in states for s in chain.latent_sequence)


def test_chain_resample_transition():
    """
    Check that resampled latent sequence conforms to deterministic transition latent
    structure
    """
    # specify starting emission sequence
    len_init = 10
    len_total = 50
    emissions = [bayesian_hmm.State("e" + str(x % len_init)) for x in range(len_total)]
    states = [bayesian_hmm.State("s" + str(x)) for x in range(8)]

    # specify deterministic latent probabilities
    p_emission = {s: {e: 1 / len(set(emissions)) for e in emissions} for s in states}
    p_transition = {s1: {s2: 1e-3 for s2 in states} for s1 in states + [bayesian_hmm.StartingState()]}
    p_transition[bayesian_hmm.StartingState()][states[0]] = 1
    for i in range(len(states)):
        p_transition[states[i]][states[(i + 1) % len(states)]] = 1 - 1e-3 * (len(states) - 1)

    # degenerate starting and transition probabilities force the given latent sequence
    latent_sequence_resampled = [states[i % len(states)] for i in range(len(emissions))]

    # initialise chain with single emission state
    chain = bayesian_hmm.Chain(emissions)
    chain.initialise(set(states))
    assert all(s in states for s in chain.latent_sequence)

    # resample latent series (several resampling steps may be required)
    for _ in range(50):
        chain.resample(set(states), p_emission, p_transition)

    # check that engineered latent structure holds
    assert list(chain.latent_sequence) == list(latent_sequence_resampled)


def test_chain_resample_emission():
    """Check that resampled latent sequence conforms to deterministic emission structure."""
    # specify starting emission sequence
    len_init = 10
    emissions = [bayesian_hmm.State("e" + str(x)) for x in range(len_init)]
    states = [bayesian_hmm.State("s" + str(x)) for x in range(2 * len_init)]

    # specify deterministic latent probabilities
    p_emission = {s: {emissions[i]: 1 if s == states[i] else 0 for i in range(len_init)} for s in states}
    p_transition = {s1: {s2: 1 / len(states) for s2 in states} for s1 in states + [bayesian_hmm.StartingState()]}

    # degenerate starting and transition probabilities force the given latent sequence
    emission_sequence = emissions * len_init
    latent_sequence_resampled = [states[i % len_init] for i in range(len(emission_sequence))]

    # initialise chain with single emission state
    chain = bayesian_hmm.Chain(emission_sequence)
    chain.initialise(set(states))
    assert all(s in states for s in chain.latent_sequence)

    # resample latent series
    for _ in range(50):
        chain.resample(set(states), p_emission, p_transition)

    # check that engineered latent structure holds
    assert all(chain.latent_sequence[x] == latent_sequence_resampled[x] for x in range(len(chain)))
