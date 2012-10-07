import sys, os
import argparse
sys.path.insert(0, os.path.abspath('..'))
from keydetection import *
from glob import glob
import pickle
import numpy as np

logging.basicConfig(level = logging.INFO)
Cache.set_caching_read(False)

def evaluate(filenames_file, models_dir):

    filenames = []
    with open(filenames_file) as f:
        for line in f.readlines():
            filenames.append(line.split('::'))

    cache = Cache('aggmodel')
    if cache.exists():
        model = cache.get()
    else:
        model_files = glob(models_dir + '/*')
        models = []
        for model_file in model_files:
            with open(model_file, 'r') as f:
                models.append(pickle.load(f))
        model = Profile.aggregate_multiple(models)
        cache.set(model)

    with open('/tmp/aggregate.pkl', 'wb') as f:
        pickle.dump(model, f)

    normalise_model(model)

    scoreboard = Scoreboard()
    for mp3_file, lab_file in filenames:
        lab = get_key_lab(lab_file)

        # songs with multiple keys is very hard. too hard for
        # this challenge.
        if lab.key_count() > 1:
            logging.warning('\n%s has more than one key (%s), skipping' % (lab_file, ', '.join(map(repr, lab.real_keys()))))
            continue

        actual_key = lab.majority_key()

        if actual_key is None:
            continue

        try:
            test_profile = get_test_profile(mp3_file, time_limit = 30, n = 4)
            #test_profile = get_test_profile(mp3_file, n = 2)

            if np.sum(test_profile.values) == 0:
                logging.warning('Silent mp3: %s' % (mp3_file))
                continue

            key = get_key(model, test_profile)
            diff = actual_key.compare(key)
            logging.info('%s: Predicted: %s; Actual: %s; Diff: %s' % (mp3_file, key, actual_key, diff.name()))
            scoreboard.add(diff)
        except Exception, e:
            logging.warning('%s: Failed to test: %s' % (mp3_file, e))

    scoreboard.print_scores()

    return scoreboard
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'evaluate')
    parser.add_argument('file')
    parser.add_argument('models_dir')
    args = parser.parse_args()

    evaluate(args.file, args.models_dir)
