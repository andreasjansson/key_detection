import util
from mrjob.job import MRJob
import httplib

def http_status(domain, path):
    conn = httplib.HTTPConnection(domain)
    conn.request('HEAD', path)
    response = conn.getresponse()
    conn.close()
    return response.status

class MREvalutator(MRJob):

    def mapper(self, _, filename):
        mp3, lab = filename.split('::')
        
        yield mp3, util.download('http://s3.amazonaws.com/andreasjansson%s' % mp3, '.mp3')

    def reducer(self, status, mp3s):
        yield status, [mp3 for mp3 in mp3s]

if __name__ == '__main__':
    MREvalutator.run()
