import sys
import cPickle

def mr_status(message):
    sys.stderr.write('reporter:status:%s\n' % message)

def mr_counter(group, counter, amount = 1):
    sys.stderr.write('reporter:counter:%s,%s,%d\n' % (group, counter, amount))

def mr_encode(data):
    return cPickle.dumps(data).encode('string_escape')

def mr_decode(line):
    return cPickle.loads(line.decode('string_escape'))
