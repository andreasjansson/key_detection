import operator
import math
import os.path
from glob import glob
import urllib
import random
import re
import tempfile
import logging

note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def make_local(filename, local_name):
    match = re.search(r'^https://s3.amazonaws.com/([^/]+)/(.+)$', filename)
    if match:
        logging.debug('Downloading %s' % (filename))
        tmp = s3_download(match.group(1), match.group(2))
        logging.debug('Downloaded %s to %s' % (filename, tmp))
        os.rename(tmp, local_name)
        logging.debug('Renamed %s to %s' % (tmp, logging))
        return local_name
    return filename

def filenames_from_twin_directories(mp3_root, lab_root, limit = None, group_by_dir = False):
    '''
    Requires the mp3_root and lab_root to have the exact same structure, with
    identical filenames, except for the file extension
    '''
    mp3_folders = set(os.listdir(mp3_root))
    lab_folders = set(os.listdir(lab_root))
    shared_folders = mp3_folders.intersection(lab_folders)

    if group_by_dir:
        filenames = {}
    else:
        filenames = []

    i = 0
    for folder in shared_folders:

        if group_by_dir:
            filenames_in_dir = []

        mp3_folder = mp3_root + "/" + folder
        lab_folder = lab_root + "/" + folder
        mp3_files = set(map(lambda s: os.path.basename(s).replace(".mp3", ""),
                            glob(mp3_folder + "/*.mp3")))
        lab_files = set(map(lambda s: os.path.basename(s).replace(".lab", ""),
                            glob(lab_folder + "/*.lab")))

        shared_files = mp3_files.intersection(lab_files)

        for f in shared_files:

            if limit is not None and i >= limit:
                return filenames

            mp3_file = mp3_folder + "/" + f + ".mp3"
            lab_file = lab_folder + "/" + f + ".lab"
            i += 1

            if group_by_dir:
                filenames_in_dir.append((mp3_file, lab_file))
            else:
                filenames.append((mp3_file, lab_file))

        if group_by_dir:
            filenames[folder] = filenames_in_dir

    return filenames

def dot_product(a, b):
    return sum(map(operator.mul, a, b))

def download(filename, suffix = ''):
    tmp = tempfile.NamedTemporaryFile(suffix = suffix, delete = False).name
    response = urllib.urlretrieve(filename, tmp)
    return tmp

def s3_upload(bucket_name, local_filename, s3_filename):
    import boto.s3.key
    import boto.s3.bucket

    conn = boto.connect_s3()
    bucket = boto.s3.bucket.Bucket(conn, bucket_name)
    key = boto.s3.key.Key(bucket)
    key.key = s3_filename
    key.set_contents_from_filename(local_filename)
    key.make_public()
    

def s3_download(bucket_name, s3_filename):
    import boto.s3.key
    import boto.s3.bucket

    local_file = tempfile.NamedTemporaryFile(suffix = os.path.splitext(s3_filename)[1], delete = False)

    conn = boto.connect_s3()
    bucket = boto.s3.bucket.Bucket(conn, bucket_name)
    k = boto.s3.key.Key(bucket)
    k.key = s3_filename
    k.get_contents_to_file(local_file)
    local_file.close()
    return local_file.name

def split_filenames(filenames, split_percent = 50, limit = None, overlap = False,
                    shuffle = True):

    if shuffle:
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
