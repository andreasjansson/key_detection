import util
from mrjob.job import MRJob
from mrjob.protocol import RawValueProtocol, PickleValueProtocol
from boto.s3.connection import S3Connection
from boto.s3.key import Key
import os
import logging
import mrjob.util

#mrjob.util.log_to_stream(name='mrjob', debug = True)
#logger = logging.getLogger('mrjob')

class MRTrain(MRJob):

    INPUT_PROTOCOL = RawValueProtocol
    INTERNAL_PROTOCOL = PickleValueProtocol
    OUTPUT_PROTOCOL = PickleValueProtocol

    def mapper(self, _, filename):

        mp3, lab = filename.split('::')

        self.set_status('Will process %s' % mp3)
        self.increment_counter('mappers', 'started')

        conn = S3Connection()
        bucket = conn.get_bucket('andreasjansson')
        mp3 = util.s3_download(bucket, mp3)
        lab = util.s3_download(bucket, lab)

        self.set_status('Downloaded to %s and %s' % (mp3, lab))

        matrices = util.get_training_matrices(mp3, lab)

        self.increment_counter('mappers', 'finished')

        os.unlink(mp3)
        os.unlink(lab)

        yield _, matrices

    def reducer(self, _, matrices_list):
        matrices = util.aggregate_matrices(matrices_list)

        yield _, matrices

if __name__ == '__main__':
    MRTrain.run()
