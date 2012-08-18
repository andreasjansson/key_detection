import sys, os
import argparse
sys.path.insert(0, os.path.abspath('..'))
from keydetection import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'bigdata')
    parser.add_argument('remote')
    parser.add_argument('local')
    args = parser.parse_args()

    tmp = s3_download('andreasjansson', args.remote)
    os.rename(tmp, args.local)
