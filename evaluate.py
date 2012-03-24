from util import *
from algorithms import *
import matplotlib.pyplot as plt
import argparse
from pprint import pprint

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
        keys_keys = [k.key for k in keys]
        keys_time = [k.time for k in keys]
        true_keys_keys = [k.key for k in true_keys]
        true_keys_time = [k.time for k in true_keys]

        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        axes.step(true_keys_time, true_keys_keys, "g-", linewidth = 2, where = 'post')
        axes.step(keys_time, keys_keys, "r-", where = 'post')
        axes.set_ylim([-1, 12]) # hack
        axes.set_yticks(range(-1, 12))
        axes.set_yticklabels(['', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', ''])
        plt.show()
        
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate key detection algorithm.')
    parser.add_argument("--algorithm", "-a", default = "FixedWindow", choices = ["FixedWindow", "BeatWindowsSimple", "BasicHMM", "GaussianMixtureHMM"])
    parser.add_argument("--plot", "-p", action = "store_true")
    parser.add_argument("--length", "-l")
    parser.add_argument("--options", "-o")
    parser.add_argument("mp3")
    parser.add_argument("truth")
    args = parser.parse_args()

    options = None
    if args.options is not None:
        options = [s.strip() for s in args.options.split(",")]
        options = dict([tuple(o.split("=")) for o in options])

    algo_class = globals()[args.algorithm]
    if args.length:
        algorithm = algo_class(args.mp3, length = float(args.length), options = options)
    else:
        algorithm = algo_class(args.mp3, options = options)
    algorithm.execute()
    keys = algorithm.filter_repeated_keys()
    parser = LabParser()
    true_keys = parser.parse_keys(args.truth)
    if args.length:
        true_keys = filter(lambda k: k.time < float(args.length), true_keys)

    evaluator = Evaluator()
    print(evaluator.evaluate(keys, true_keys))

    if args.plot:
        if args.length:
            max_length = float(args.length)
        else:
            max_length = len(algorithm.audio) / algorithm.samp_rate + 1 # +1 for rounding errors
        def pad_keys(keys):
            return keys + [Key(keys[-1].key, max_length)]
        evaluator.plot(pad_keys(keys), pad_keys(true_keys))
    
