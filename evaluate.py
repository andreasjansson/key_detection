from naive import Naive
from util import *
import argparse

class Evaluator:
    """
    Evaluate the effectiveness of a key detection algortihm.
    """

    def evaluate(self, keys, true_keys, time_window = 1):

        i = j = 0
        real_hits = trans_hits = key_hits = misses = incorrect = 0

        while i < len(keys) and j < len(true_keys):

            key = keys[i]
            true_key = true_keys[j]

            if abs(key.time - true_key.time) <= time_window:
                if key.key == true_key.key:
                    real_hits += 1
                else:
                    trans_hits += 1
                i += 1
                j += 1
                
            elif key.time < true_key.time:
                # first keys should both be at time 0, so j - 1 should
                # never be -1.
                if key.key == true_keys[j - 1].key:
                    key_hits += 1
                else:
                    incorrect += 1
                i += 1

            else:
                j += 1
                misses += 1

        if i < len(keys):

            if not true_key:
                true_key = true_keys[0]

            for i in range(i, len(keys)):
                if keys[i].key == true_key.key:
                    key_hits += 1
                else:
                    incorrect += 1

        if j < len(true_keys):
            misses += len(true_keys) - j

        return {'real_hits': real_hits, 'trans_hits': trans_hits,
                'key_hits': key_hits, 'incorrect': incorrect,
                'misses': misses}

    def plot(self, keys, true_keys):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate key detection algorithm.')
    parser.add_argument("--algorithm", "-a", default = "Naive", choices = ["Naive"])
    parser.add_argument("--plot", "-p", action = "store_true", nargs = 0)
    parser.add_argument("mp3")
    parser.add_argument("truth")
    args = parser.parse_args()

    algorithm = globals()[args.algorithm](args.mp3)
    keys = algorithm.execute()
    parser = LabParser()
    true_keys = parser.parse_keys(args.truth)

    evaluator = Evaluator()
    print(evaluator.evaluate(keys, true_keys))

    if args.plot:
        evaluator.plot(keys, true_keys)
    
