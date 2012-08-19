import sys, os
import argparse
sys.path.insert(0, os.path.abspath('..'))
from keydetection import *
from glob import glob
import pickle

logging.basicConfig(level = logging.INFO)
Cache.set_caching_enabled(False)

def aggregate(models_dir, local_model, output_filename):

    model_files = glob(models_dir + '/*') + [local_model]
    models = []
    for model_file in model_files:
        with open(model_file, 'r') as f:
            models.append(pickle.load(f))
    model = aggregate_matrices(models)

    with open(output_filename, 'wb') as f:
        pickle.dump(model, f)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'aggregate')
    parser.add_argument('models')
    parser.add_argument('local')
    parser.add_argument('output')
    args = parser.parse_args()

    aggregate(args.models, args.local, args.output)
