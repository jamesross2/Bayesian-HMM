import bayesian_hmm.utils


def test_label_generator() -> None:
    label_generator = bayesian_hmm.utils.label_generator(["a", "b", "c"])
    labels = [next(label_generator) for _ in range(10)]
    assert labels == ["a", "b", "c", "a1", "b1", "c1", "a2", "b2", "c2", "a3"]


def test_dirichlet_process_generator() -> None:
    good_generator = bayesian_hmm.utils.dirichlet_process_generator()
    outputs = [next(good_generator) for _ in range(100)]
    assert len(outputs) == 100
    assert len(set(outputs)) < 100

    # try a generator that should rarely create a new label
    bad_generator = bayesian_hmm.utils.dirichlet_process_generator(alpha=1e-8)
    outputs = [next(bad_generator) for _ in range(100)]
    assert len(outputs) == 100
    assert len(set(outputs)) < 5


def test_max_dict() -> None:
    eps = 1e-12
    d_orig = {"a": 1, "b": 1e6, "c": 1e-16}
    d_max = bayesian_hmm.utils.max_dict(d_orig, eps=eps)
    assert d_max["a"] == d_orig["a"]
    assert d_max["b"] == d_orig["b"]
    assert d_max["c"] != d_orig["a"]
    assert d_max["c"] == 1e-12


def test_shrinkage() -> None:
    d_orig = {"a": 0.2, "b": 0.3, "c": 0.5}
    d_shrunk = bayesian_hmm.utils.shrink_probabilities(d_orig)
    assert d_shrunk["a"] > d_orig["a"]
    assert d_shrunk["b"] > d_orig["b"]
    assert d_shrunk["c"] < d_orig["c"]
    assert d_shrunk["a"] < d_shrunk["b"] < d_shrunk["c"]

    # check that tuple produces identical results
    t_orig = tuple(d_orig.values())
    t_shrunk = bayesian_hmm.utils.shrink_probabilities(t_orig)
    assert tuple(d_shrunk.values()) == t_shrunk
