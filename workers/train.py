import sys, os
import argparse
sys.path.insert(0, os.path.abspath('..'))
from keydetection import *

logging.basicConfig(level = logging.INFO)

def train(filenames_file, model_s3_filename):

    s3_delete('andreasjansson', model_s3_filename)

    filenames = []
    with open(filenames_file) as f:
        for line in f.readlines():
            filenames.append(line.split('::'))

    model = get_aggregate_markov_matrices(filenames)
    model_local_filename = '/tmp/model.pkl'
    with open(model_local_filename, 'wb') as f:
        pickle.dump(model, f)
    
    s3_upload('andreasjansson', model_local_filename, model_s3_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'bigdata')
    parser.add_argument('file')
    parser.add_argument('model')
    args = parser.parse_args()

    evaluate(args.file, args.model)

