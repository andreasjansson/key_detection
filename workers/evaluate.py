import sys, os
import argparse
sys.path.insert(0, os.path.abspath('..'))
from keydetection import *
from glob import glob

logging.basicConfig(level = logging.INFO)

def evaluate(filenames_file, models_dir):

    filenames = []
    with open(filenames_file) as f:
        for line in f.readlines():
            filenames.append(line.split('::'))

    model_files = glob(models_dir + '/*')
    models = []
    for model_file in model_files:
        with open(model_file, 'r') as f:
            models.append(pickle.load(f))
    model = aggregate_matrices(models)

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

        print '\nTesting ' + mp3_file
        try:
            test_matrix = get_test_matrix(mp3_file)
            key = get_key(model, test_matrix, unmarkov = True)
            diff = actual_key.compare(key)
            print 'Predicted: %s; Actual: %s; Diff: %s' % (key, actual_key, diff.name())
            scoreboard.add(diff)
        except Exception:
            print 'Failed to test %s' % mp3_file

    scoreboard.print_scores()

    return scoreboard
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'bigdata')
    parser.add_argument('file')
    parser.add_argument('models_dir')
    args = parser.parse_args()

    evaluate(args.file, args.models_dir)
