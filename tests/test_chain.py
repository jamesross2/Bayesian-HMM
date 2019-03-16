from unittest import TestCase

import bayesian_hmm


class TestChain(TestCase):
    def test_stores_emissions(self):
        seq = ['a', 'b', 'c']
        chain = bayesian_hmm.Chain(seq)
        self.assertTrue(chain.T == 3)
        self.assertTrue(chain.emission_sequence == ['a', 'b', 'c'])
