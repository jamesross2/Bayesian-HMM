import scipy.stats

import bayesian_hmm


def test_model_initialisation() -> None:
    model = bayesian_hmm.HierarchicalDirichletProcess()
    assert isinstance(model, bayesian_hmm.HierarchicalDirichletProcess)
    assert isinstance(model, bayesian_hmm.Variable)

    # TODO: check that provided priors for variables are honoured


def test_model_likelihood() -> None:
    # set alpha so that stick breaking process beta variables have a pdf strictly below 1
    model = bayesian_hmm.HierarchicalDirichletProcess(
        alpha_prior=lambda: 1.0, alpha_log_likelihood=lambda x: scipy.stats.gamma.logpdf(x=x, a=1, scale=1)
    )
    likelihood_init = model.log_likelihood()
    assert likelihood_init < 0
    assert model.log_likelihood() == model.alpha.log_likelihood() + model.gamma.log_likelihood()

    # check that likelihood updates as variables are added
    num_symbols = 10
    likelihoods = [model.log_likelihood()]
    for val in range(num_symbols):
        model.add_state(bayesian_hmm.State(val))
        likelihoods.append(model.log_likelihood())
    assert len(set(likelihoods)) > 1


def test_model_resampling() -> None:
    # TODO: add states to model
    states = {bayesian_hmm.State(x) for x in {"a", "char", 5}}
    states.add(bayesian_hmm.AggregateState())
    counts = {state0: {state1: 3 for state1 in states} for state0 in states}

    for sticky in (True, False):
        model = bayesian_hmm.HierarchicalDirichletProcess(sticky=sticky)
        _ = [model.add_state(state) for state in states if state != bayesian_hmm.AggregateState()]
        model.resample(counts=counts)
