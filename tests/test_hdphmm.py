import bayesian_hmm


def test_empty_hmm():
    emission_sequences = []
    hmm = bayesian_hmm.HierarchicalDirichletProcessHiddenMarkovModel(emission_sequences)
    assert not hmm.initialised
    hmm.initialise(0)
    assert hmm.initialised
    assert hmm.emissions == set()
    assert hmm.c == 0
    assert hmm.k == 0
    assert hmm.n == 0
    
    
def test_print():
    # checks that printing does not cause an error
    emission_sequences = [['e' + str(x) for x in range(l)] for l in range(1, 50)]
    hmm = bayesian_hmm.HierarchicalDirichletProcessHiddenMarkovModel(emission_sequences)
    hmm.initialise(20)
    print(hmm)
    repr(hmm)
    hmm.print_fit_parameters()
    hmm.print_probabilities()
    assert True


def test_initialise_hmm():
    emission_sequences = [['e' + str(x) for x in range(l)] for l in range(1, 50)]
    hmm = bayesian_hmm.HierarchicalDirichletProcessHiddenMarkovModel(emission_sequences)
    hmm.initialise(20)
    
    # check chain initialised correctly
    assert hmm.emissions.symmetric_difference(set(['e' + str(x) for x in range(49)])) == set()
    assert hmm.c == 49
    assert hmm.k == 20
    assert hmm.n == 49
