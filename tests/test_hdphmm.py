import bayesian_hmm


def test_empty_hmm():
    emission_sequences = []
    hmm = bayesian_hmm.HDPHMM(emission_sequences)
    assert not hmm.initialised
    hmm.initialise(0)
    assert hmm.initialised
    assert hmm.emissions == set()
    assert hmm.c == 0
    assert hmm.k == 0
    assert hmm.n == 0


def test_print():
    # checks that printing does not cause an error
    emission_sequences = [["e" + str(x) for x in range(l)] for l in range(1, 50)]
    hmm = bayesian_hmm.HDPHMM(emission_sequences)
    hmm.initialise(20)
    print(hmm)
    repr(hmm)
    hmm.print_fit_parameters()
    hmm.print_probabilities()
    assert True


def test_initialise_hmm():
    emission_sequences = [["e" + str(x) for x in range(l)] for l in range(1, 50)]
    hmm = bayesian_hmm.HDPHMM(emission_sequences)
    hmm.initialise(20)

    # check chain initialised correctly
    assert (
        hmm.emissions.symmetric_difference(set(["e" + str(x) for x in range(49)]))
        == set()
    )
    assert hmm.c == 49
    assert hmm.k == 20
    assert hmm.n == 49


def test_sticky_initialisation():
    emission_sequences = [[1, 2, 3] * 3] * 3
    hmm_sticky = bayesian_hmm.HDPHMM(emission_sequences, sticky=True)
    hmm_slippery = bayesian_hmm.HDPHMM(emission_sequences, sticky=False)
    hmm_sticky.initialise(20)
    hmm_slippery.initialise(20)

    # check chain initialises correctly in both cases
    assert 0 <= hmm_sticky.hyperparameters["kappa"] <= 1
    assert hmm_sticky.priors["kappa"] != (lambda: 0)
    assert hmm_slippery.hyperparameters["kappa"] == 0
    assert all(hmm_slippery.priors["kappa"]() == 0 for _ in range(100))


def test_manual_priors():
    emission_sequences = [[1, 2, 3] * 3] * 3
    priors_default = {
        "alpha": lambda: np.random.gamma(2, 2),
        "gamma": lambda: np.random.gamma(3, 3),
        "alpha_emission": lambda: np.random.gamma(2, 2),
        "gamma_emission": lambda: np.random.gamma(3, 3),
        "kappa": lambda: np.random.beta(1, 1),
    }
    hmms = {
        "default": bayesian_hmm.HDPHMM(emission_sequences),
        "single": bayesian_hmm.HDPHMM(emission_sequences, priors={"alpha": lambda: -1}),
        "all": bayesian_hmm.HDPHMM(
            emission_sequences,
            priors={param: lambda: -1 for param in priors_default.keys()},
        ),
    }

    # check that priors work in all cases
    assert all(param > 0 for param in hmms["default"].hyperparameters.values())
    assert all(param < 0 for param in hmms["all"].hyperparameters.values())
    assert hmms["single"].hyperparameters["alpha"] < 0
    assert all(
        hmms["single"].hyperparameters[param] > 0
        for param in priors_default.keys()
        if param != "alpha"
    )

    fail = False
    try:
        _ = bayesian_hmm.HDPHMM(
            emission_sequences, priors={"kappa": lambda: 2}, sticky=False
        )
    except ValueError:
        fail = True
    assert fail
