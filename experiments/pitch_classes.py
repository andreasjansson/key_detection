# Path hack.
import sys, os
sys.path.insert(0, os.path.abspath('..'))

# usage:
#   python pitch_classes.py mp3dir labdir
#
# e.g.:
#   python pitch_classes.py /home/andreas/music/The\ Beatles /home/andreas/data/beatles_annotations/keylab/the_beatles

from keydetection import *

filenames = [(mp3, lab) for mp3, lab in
             filenames_from_twin_directories(sys.argv[1], sys.argv[2])]

training, testing = split_filenames(filenames, limit = 2)

markov_model = get_aggregate_markov_matrices(training)
pitch_class_model = get_pitch_class_model()

markov_scoreboard = Scoreboard()
pitch_class_scoreboard = Scoreboard()

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

    test_matrix = get_test_matrix(mp3_file)

    markov_key = get_key(markov_model, test_matrix)
    markov_diff = actual_key.compare(markov_key)

    pitch_class_key = get_key(pitch_class_model, test_matrix)
    pitch_class_diff = actual_key.compare(pitch_class_key)

    print 'Markov Predicted:      %s; Actual: %s; Diff: %s' % (markov_key, actual_key, markov_diff.name())
    print 'Pitch Class Predicted: %s; Actual: %s; Diff: %s' % (pitch_class_key, actual_key, pitch_class_diff.name())
    markov_scoreboard.add(markov_diff)
    pitch_class_scoreboard.add(pitch_class_diff)

markov_scoreboard.print_scores()
pitch_class_scoreboard.print_scores()
