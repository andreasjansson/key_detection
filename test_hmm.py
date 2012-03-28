import unittest
from util import *

class HMMTest(unittest.TestCase):

    def testViterbi(self):
        trans_probs = [[0.6, 0.4],
                       [0.1, 0.9]]
        profiles = [[1, 0], [0, 1]]
        start_probs = [.5, .5]
        hmm = HMM(profiles, trans_probs, start_probs)
        
        emissions = [[.6, .2], [.5, .5]]
        path = hmm.viterbi(emissions)
        self.assertEqual([0, 0], path)
        
        emissions = [[.6, .2], [.3, .7]]
        path = hmm.viterbi(emissions)
        self.assertEqual([0, 1], path)
        
        emissions = [[.6, .2], [.5, .5], [.6, .4]]
        path = hmm.viterbi(emissions)
        self.assertEqual([0, 0, 0], path)
        
        emissions = [[.6, .2], [.5, .5], [.4, .6]]
        path = hmm.viterbi(emissions)
        self.assertEqual([0, 1, 1], path)

if __name__ == '__main__':
    unittest.main()
