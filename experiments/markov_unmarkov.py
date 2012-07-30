# Path hack.
import sys, os
sys.path.insert(0, os.path.abspath('..'))

# usage:
#   python markov_unmarkov.py mp3dir labdir
#
# e.g.:
#   python markov_unmarkov.py /home/andreas/music/The\ Beatles /home/andreas/data/beatles_annotations/keylab/the_beatles

import sys
from keydetection import *

filenames = [(mp3, lab) for mp3, lab in
             filenames_from_twin_directories(sys.argv[1], sys.argv[2])]
training, testing = split_filenames(filenames, limit = 10, shuffle = True)

print len(training), len(testing)

markov_model = get_aggregate_markov_matrices(training)
unmarkov_model = [None] * 24
for i in range(24):
    unmarkov_model[i] = markov_model[i].get_unmarkov_array()

markov_scoreboard = Scoreboard()
unmarkov_scoreboard = Scoreboard()

for mp3_file, lab_file in testing:
    lab = KeyLab(lab_file)

    # songs with multiple keys is very hard. too hard for
    # this challenge.
    if lab.key_count() > 1:
        print '\n%s has more than one key (%s), skipping' % (lab_file, ', '.join(map(repr, lab.real_keys())))
        continue

    actual_key = lab.majority_key()

    if actual_key is None:
        continue

    print '\nTesting ' + mp3_file

    test_markov_matrix = get_test_matrix(mp3_file)
    markov_key = get_key(markov_model, test_markov_matrix)
    markov_diff = actual_key.compare(markov_key)

    test_unmarkov = test_markov_matrix.get_unmarkov_array()
    unmarkov_key = get_key_from_unmarkov_array(unmarkov_model, test_unmarkov)
    unmarkov_diff = actual_key.compare(unmarkov_key)

    print 'Markov Predicted:      %s; Actual: %s; Diff: %s' % (markov_key, actual_key, markov_diff.name())
    print 'Unmarkov Predicted: %s; Actual: %s; Diff: %s' % (unmarkov_key, actual_key, unmarkov_diff.name())
    markov_scoreboard.add(markov_diff)
    unmarkov_scoreboard.add(unmarkov_diff)

markov_scoreboard.print_scores()
unmarkov_scoreboard.print_scores()
