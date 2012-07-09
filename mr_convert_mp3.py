from mrjob.job import MRJob
import logging
import mrjob.util
from boto.s3.connection import S3Connection
from boto.s3.key import Key
import os
import os.path
import tempfile
import subprocess
import urllib

class MRConvertMP3(MRJob):

    def mapper(self, _, line):

        mrjob.util.log_to_stream(name='mrjob.emr', debug = True)
        logger = logging.getLogger('mrjob.emr')

        conn = S3Connection()
        bucket = conn.get_bucket('andreasjansson')

        s3_ly_file, s3_midi_file = line.split('::')

        midi_file = tempfile.NamedTemporaryFile(suffix = '.wav', delete = False).name
        wav_file = tempfile.NamedTemporaryFile(suffix = '.wav', delete = False).name
        mp3_file = tempfile.NamedTemporaryFile(suffix = '.mp3', delete = False).name

        k = Key(bucket)
        k.key = s3_midi_file
        k.get_contents_to_filename(midi_file)

        logger.info(subprocess.Popen(
                ['timidity', '-Ow', '-o' + wav_file, midi_file],
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE)
              .communicate()[0])

        logger.info(subprocess.Popen(
                ['lame', '-b128', wav_file, mp3_file],
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE)
              .communicate()[0])

        os.unlink(wav_file)

        s3_mp3_file = 'lilymp3/%s.mp3' % os.path.basename(s3_ly_file)

        k = Key(bucket)
        k.key = s3_mp3_file
        k.set_contents_from_filename(mp3_file)

        logger.info('Uploaded to http://s3.amazonaws.com/andreasjansson/%s' % s3_mp3_file)

        os.unlink(mp3_file)

        yield s3_ly_file, s3_mp3_file

    def reducer(key, value):
        yield key, value

if __name__ == '__main__':
    MRConvertMP3.run()
