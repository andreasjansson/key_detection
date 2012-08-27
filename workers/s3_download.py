import sys, os
import argparse
sys.path.insert(0, os.path.abspath('..'))
from keydetection import *
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 's3download')
    parser.add_argument('remote')
    parser.add_argument('local')
    args = parser.parse_args()

    remote = args.remote
    remote = re.sub('^https:\/\/s3\.amazonaws\.com/andreasjansson/', '', remote)
    print remote

    tmp = s3_download('andreasjansson', remote)
    os.rename(tmp, args.local)
