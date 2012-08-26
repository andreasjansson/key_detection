import operator
import math
import os.path
from glob import glob
import urllib
import random
import re
import tempfile
import logging

from key import *

note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def get_key(model, test_profile):
    '''
    Computes the key based on a trained model and
    a test profile.
    '''
    argmax = -1
    maxsim = 0
    for i, profile in enumerate(model):

        sim = profile.similarity(test_profile)

        if sim > maxsim:
            maxsim = sim
            argmax = i

    argmax = argmax % 24
    if argmax < 12:
        return MajorKey(argmax)
    else:
        return MinorKey(argmax - 12)

def download(filename, suffix = ''):
    '''
    Download a file from the Internet to a temporary location. Return
    the filename of the downloaded file.
    '''
    tmp = tempfile.NamedTemporaryFile(suffix = suffix, delete = False)
    tmp.close()
    response = urllib.urlretrieve(filename, tmp.name)
    return tmp.name

def make_local(filename, local_name):
    match = re.search(r'^https?://', filename)
    if match:
        logging.debug('Downloading %s' % (filename))
        tmp = download(filename)
        logging.debug('Downloaded %s to %s' % (filename, tmp))
        os.rename(tmp, local_name)
        logging.debug('Renamed %s to %s' % (tmp, local_name))
        return local_name
    return filename
