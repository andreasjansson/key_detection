import sys, os
import argparse
sys.path.insert(0, os.path.abspath('..'))
from keydetection import *

logging.basicConfig(level = logging.INFO)

def train(filenames_file, model_filename):
    filenames = []
    with open(filenames_file) as f:
        for line in f.readlines():
            filenames.append(line.split('::'))

    model = get_aggregate_markov_matrices(filenames)
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'bigdata')
    parser.add_argument('file')
    parser.add_argument('model')
    args = parser.parse_args()

    train(args.file, args.model)

