import util
import argparse
import os
import os.path as path
from glob import glob
import pickle

def train(mp3dir, labdir, limit = None):
    training_filenames = filenames(mp3dir, labdir, limit)
    training_matrices = util.get_aggregate_markov_matrices(training_filenames)
    return training_matrices

def filenames(mp3_root, lab_root, limit = None):
    mp3_folders = set(os.listdir(mp3_root))
    lab_folders = set(os.listdir(lab_root))
    shared_folders = mp3_folders.intersection(lab_folders)

    i = 0
    for folder in shared_folders:

        mp3_folder = mp3_root + "/" + folder
        lab_folder = lab_root + "/" + folder
        mp3_files = set(map(lambda s: path.basename(s).replace(".mp3", ""),
                            glob(mp3_folder + "/*.mp3")))
        lab_files = set(map(lambda s: path.basename(s).replace(".lab", ""),
                            glob(lab_folder + "/*.lab")))

        shared_files = mp3_files.intersection(lab_files)

        for f in shared_files:

            if limit is not None and i >= limit:
                raise StopIteration

            mp3_file = mp3_folder + "/" + f + ".mp3"
            lab_file = lab_folder + "/" + f + ".lab"
            i += 1
            yield (lab_file, mp3_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train key detector')
    parser.add_argument('-m', '--mp3dir', required = True)
    parser.add_argument('-l', '--labdir', required = True)
    parser.add_argument('-o', '--outfile', default = 'model.pkl')
    parser.add_argument('--limit')
    args = parser.parse_args()
    matrices = train(args.mp3dir, args.labdir, int(args.limit))
    with open(args.outfile, 'wb') as outfile:
        pickle.dump(matrices, outfile)
