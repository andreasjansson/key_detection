import util
from boto.s3.connection import S3Connection
from boto.s3.key import Key
import os
import os.path
import logging
import sys
import cPickle

sys.path.append(os.path.dirname(__file__))

def mapper(local_dir = None):

    if local_dir is None:
        conn = S3Connection()
        bucket = conn.get_bucket('andreasjansson')

    for line in sys.stdin:
        line = line.split('\n')[0]
        mp3, lab = line.split('::')

        util.mr_status('Will process %s' % mp3)
        util.mr_counter('mappers', 'started')

        if local_dir:
            mp3 = local_dir + mp3
            lab = local_dir + lab
            if not os.path.exists(mp3) or not os.path.exists(lab):
                continue
        else:
            try:
                mp3 = util.s3_download(bucket, mp3)
                lab = util.s3_download(bucket, lab)
            except Exception as e:
                util.mr_status(str(e))
                continue

            util.mr_status('Downloaded to %s and %s' % (mp3, lab))

        matrices = util.get_training_matrices(mp3, lab)

        util.mr_counter('mappers', 'finished')

        if local_dir is None:
            os.unlink(mp3)
            os.unlink(lab)

        print util.mr_encode(matrices)

def reducer(local_dir = None):

    aggregate_matrices = [util.MarkovMatrix(12 * 12) for i in range(12 * 2)]
    for line in sys.stdin:
        line = line.split('\n')[0]

        matrices = util.mr_decode(line)

        if matrices is not None:
            for i in range(24):
                aggregate_matrices[i].add(matrices[i])

    for matrix in aggregate_matrices:
        matrix.normalise()

    print util.mr_encode(aggregate_matrices)

if __name__ == '__main__':

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--localdir', '-l')
    parser.add_argument('action', choices = ['mapper', 'reducer'])
    args = parser.parse_args()
    globals()[args.action](args.localdir)
 
