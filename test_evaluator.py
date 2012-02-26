from util import Key
from evaluate import Evaluator
import unittest

class TestEvaluator(unittest.TestCase):

    def test_evaluate(self):
        e = Evaluator()

        self.assertEqual(
            {'real_hits':  1,
             'trans_hits': 0,
             'key_hits'  : 0,
             'misses'    : 0,
             'incorrect' : 0},
            e.evaluate([Key('a', 0)], [Key('a', 0)]))

        self.assertEqual(
            {'real_hits':  0,
             'trans_hits': 1,
             'key_hits'  : 0,
             'misses'    : 2,
             'incorrect' : 0},
            e.evaluate([Key('a', 0)],
                       [Key('b', 0), Key('b', 1),
                        Key('c', 2)]))

        self.assertEqual(
            {'real_hits':  0,
             'trans_hits': 1,
             'key_hits'  : 1,
             'misses'    : 0,
             'incorrect' : 1},
            e.evaluate([Key('a', 0),
                        Key('b', 1),
                        Key('c', 2)],
                       [Key('b', 0)]))

        self.assertEqual(
            {'real_hits':  1,
             'trans_hits': 0,
             'key_hits'  : 0,
             'misses'    : 1,
             'incorrect' : 1},
            e.evaluate([Key('a', 0),
                        Key('e', 10)],
                       [Key('a', 0),
                        Key('d', 5)]))

        self.assertEqual(
            {'real_hits':  1,
             'trans_hits': 1,
             'key_hits'  : 1,
             'misses'    : 2,
             'incorrect' : 2},
            e.evaluate([Key('a', 0), Key('d', 2), Key('e', 6), Key('a', 8), Key('e', 14)],
                       [Key('e', 0), Key('d', 4), Key('a', 9), Key('e', 12)]))
        


if __name__ == '__main__':
    unittest.main()
