# TODO: test with different training sets;
#       test with "handcoded" model
#       why is yellow submarine tested as C# major!!!?
#       normalise matrices to 1, otherwise we'll never match minor keys
#       find collection of annotated classical music (where the
#         annotation format supports key, e.g. abc), synthesise with
#         timidity and use these as training data. compare to
#         beatles data.

import operator
import math
import os.path
#import matplotlib.pyplot as plt
from glob import glob
import urllib
import boto.s3.key
#import pdb
import random

note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']


def filenames_from_file(filename):
    '''
    Reads from a file with colon-separated (mp3_filename.mp3:lab_filename.lab)
    lines.
    '''
    # TODO
    pass

def filenames_from_twin_directories(mp3_root, lab_root, limit = None):
    '''
    Requires the mp3_root and lab_root to have the exact same structure, with
    identical filenames, except for the file extension
    '''
    mp3_folders = set(os.listdir(mp3_root))
    lab_folders = set(os.listdir(lab_root))
    shared_folders = mp3_folders.intersection(lab_folders)

    i = 0
    for folder in shared_folders:

        mp3_folder = mp3_root + "/" + folder
        lab_folder = lab_root + "/" + folder
        mp3_files = set(map(lambda s: os.path.basename(s).replace(".mp3", ""),
                            glob(mp3_folder + "/*.mp3")))
        lab_files = set(map(lambda s: os.path.basename(s).replace(".lab", ""),
                            glob(lab_folder + "/*.lab")))

        shared_files = mp3_files.intersection(lab_files)

        for f in shared_files:

            if limit is not None and i >= limit:
                raise StopIteration

            mp3_file = mp3_folder + "/" + f + ".mp3"
            lab_file = lab_folder + "/" + f + ".lab"
            i += 1
            yield (mp3_file, lab_file)

def dot_product(a, b):
    return sum(map(operator.mul, a, b))

def download(filename, suffix = ''):
    tmp = tempfile.NamedTemporaryFile(suffix = suffix, delete = False).name
    response = urllib.urlretrieve(filename, tmp)
    return tmp

def s3_download(bucket, s3_filename):
    local_file = tempfile.NamedTemporaryFile(suffix = os.path.splitext(s3_filename)[1], delete = False)
    k = boto.s3.key.Key(bucket)
    k.key = s3_filename
    k.get_contents_to_file(local_file)
    local_file.close()
    return local_file.name

def split_filenames(filenames, split_percent = 50, limit = None, overlap = False):

    random.shuffle(filenames)
    if limit:
        filenames = filenames[:limit]

    if overlap:
        first = filenames
        second = filenames
    else:
        split = int(math.ceil(len(filenames) * split_percent / 100))
        first = filenames[:split]
        second = filenames[split:]

    return (first, second)
