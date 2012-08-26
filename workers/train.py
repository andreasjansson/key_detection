import sys, os
import argparse
sys.path.insert(0, os.path.abspath('..'))
from keydetection import *

Cache.set_caching_read(False)

def train(filenames_file, model_filename, local_model = False):

    if not local_model:
        s3_delete('andreasjansson', model_filename)

    filenames = []
    with open(filenames_file) as f:
        for line in f.readlines():
            filenames.append(line.split('::'))

    model = get_trained_model(filenames)
    model_local_filename = '/tmp/model.pkl'
    with open(model_local_filename, 'wb') as f:
        pickle.dump(model, f)

    if local_model:
        import shutil
        shutil.copy(model_local_filename, model_filename)
        os.unlink(model_local_filename)
    else:
        s3_upload('andreasjansson', model_local_filename, model_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'train')
    parser.add_argument('-v', '--verbose', action = 'store_true')
    parser.add_argument('-l', '--local', action = 'store_true')
    parser.add_argument('file')
    parser.add_argument('model')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)

    train(args.file, args.model, args.local)

