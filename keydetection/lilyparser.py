import re
import sys
from glob import glob
import os.path

import key

language_notes = {'norsk': {'ges': 6, 'diss': 3, 'dessess': 0, 'aess': 8, 'gessess': 5, 'aessess': 7, 'fiss': 6, 'as': 8, 'eiss': 5, 'ais': 10, 'ceses': -2, 'bess': 9, 'fisis': 7, 'hississ': 13, 'es': 3, 'deses': 0, 'des': 1, 'eses': 2, 'cisis': 2, 'fississ': 7, 'c': 0, 'essess': 2, 'cess': -1, 'geses': 5, 'fes': 4, 'hiss': 12, 'eessess': 2, 'gisis': 9, 'aississ': 11, 'ass': 8, 'giss': 8, 'his': 12, 'cessess': -2, 'disis': 4, 'ases': 7, 'hisis': 13, 'dess': 1, 'eeses': 2, 'feses': 3, 'ces': -1, 'fess': 4, 'cississ': 2, 'aisis': 11, 'ess': 3, 'cis': 1, 'eississ': 6, 'assess': 7, 'a': 9, 'dississ': 4, 'aes': 8, 'b': 10, 'aeses': 7, 'd': 2, 'g': 7, 'f': 5, 'eisis': 6, 'h': 11, 'fessess': 3, 'eis': 5, 'ees': 3, 'aiss': 10, 'gississ': 9, 'fis': 6, 'gis': 8, 'eess': 3, 'bes': 9, 'ciss': 1, 'e': 4, 'gess': 6, 'dis': 3}, 'deutsch': {'as': 8, 'ceses': -2, 'fisis': 7, 'es': 3, 'deses': 0, 'eses': 2, 'cisis': 2, 'heses': 9, 'geses': 5, 'fes': 4, 'aisis': 11, 'gisis': 9, 'ges': 6, 'his': 12, 'asas': 7, 'fis': 6, 'c': 0, 'hisis': 13, 'feses': 3, 'ces': -1, 'gis': 8, 'a': 9, 'disis': 4, 'eis': 5, 'b': 10, 'e': 4, 'd': 2, 'g': 7, 'f': 5, 'eisis': 6, 'h': 11, 'des': 1, 'ais': 10, 'cis': 1, 'dis': 3}, 'suomi': {'as': 8, 'ceses': -2, 'fisis': 7, 'es': 3, 'deses': 0, 'eses': 2, 'cisis': 2, 'geses': 5, 'fes': 4, 'aisis': 11, 'gisis': 9, 'ges': 6, 'his': 12, 'asas': 7, 'fis': 6, 'c': 0, 'hisis': 13, 'feses': 3, 'ces': -1, 'gis': 8, 'a': 9, 'disis': 4, 'eis': 5, 'b': 10, 'e': 4, 'd': 2, 'g': 7, 'f': 5, 'eisis': 6, 'h': 11, 'des': 1, 'ais': 10, 'cis': 1, 'bes': 9, 'dis': 3}, 'espanol': {'mib': 3, 'labb': 7, 'ress': 4, 'sol': 7, 'sib': 10, 'rebb': 0, 'doss': 2, 'miss': 6, 'mibb': 2, 'sis': 12, 'rex': 4, 'la': 9, 'fass': 7, 'six': 13, 'dox': 2, 'mix': 6, 'fabb': 3, 'reb': 1, 'mis': 5, 'solss': 9, 'solbb': 5, 'do': 0, 'fax': 7, 'fas': 6, 'solb': 6, 'lab': 8, 'fa': 5, 'lax': 11, 'solx': 9, 'fab': 4, 're': 2, 'sols': 8, 'siss': 13, 'las': 10, 'lass': 11, 'dobb': -2, 'dob': -1, 'mi': 4, 'sibb': 9, 'si': 11, 'res': 3, 'dos': 1}, 'svenska': {'diss': 3, 'dessess': 0, 'giss': 8, 'assess': 7, 'fiss': 6, 'eiss': 5, 'hississ': 13, 'aississ': 11, 'ess': 3, 'fississ': 7, 'cess': -1, 'hiss': 12, 'essess': 2, 'ass': 8, 'gessess': 5, 'cessess': -2, 'dess': 1, 'fess': 4, 'cississ': 2, 'eississ': 6, 'd': 2, 'a': 9, 'dississ': 4, 'c': 0, 'b': 10, 'e': 4, 'hessess': 9, 'g': 7, 'fessess': 3, 'h': 11, 'f': 5, 'ciss': 1, 'aiss': 10, 'gississ': 9, 'gess': 6}, 'italiano': {'labb': 7, 'rebb': 0, 'sol': 7, 'sib': 10, 'ladd': 11, 'redd': 4, 'mibb': 2, 'la': 9, 'mib': 3, 'mid': 5, 're': 2, 'sid': 12, 'reb': 1, 'red': 3, 'solbb': 5, 'do': 0, 'soldd': 9, 'lad': 10, 'sold': 8, 'fadd': 7, 'solb': 6, 'lab': 8, 'fa': 5, 'fab': 4, 'fad': 6, 'fabb': 3, 'dodd': 2, 'dob': -1, 'midd': 6, 'dobb': -2, 'mi': 4, 'dod': 1, 'sibb': 9, 'si': 11, 'sidd': 13}, 'catalan': {'redd': 4, 'ladd': 11, 'ress': 4, 'soldd': 9, 'sol': 7, 'rebb': 0, 'doss': 2, 'miss': 6, 'mibb': 2, 'sis': 12, 'la': 9, 'fass': 7, 'mib': 3, 'mid': 5, 'sib': 10, 'siss': 13, 'fabb': 3, 'reb': 1, 'mis': 5, 'solss': 9, 'red': 3, 'solbb': 5, 'do': 0, 'sid': 12, 'lad': 10, 'fas': 6, 'sold': 8, 'fadd': 7, 'solb': 6, 'lab': 8, 'fa': 5, 'fab': 4, 're': 2, 'sols': 8, 'fad': 6, 'las': 10, 'lass': 11, 'dodd': 2, 'dob': -1, 'midd': 6, 'dobb': -2, 'mi': 4, 'dod': 1, 'labb': 7, 'sibb': 9, 'si': 11, 'res': 3, 'dos': 1, 'sidd': 13}, 'vlaams': {'labb': 7, 'sol': 7, 'sib': 10, 'solb': 6, 'rebb': 0, 'fakk': 7, 'mibb': 2, 'reb': 1, 'mik': 5, 'lakk': 11, 'la': 9, 'mib': 3, 'rek': 3, 'sikk': 13, 'fabb': 3, 'sik': 12, 'solbb': 5, 'do': 0, 'solk': 8, 'lak': 10, 'dokk': 2, 'mikk': 6, 'lab': 8, 'fak': 6, 'fab': 4, 're': 2, 'dok': 1, 'fa': 5, 'dob': -1, 'dobb': -2, 'mi': 4, 'solkk': 9, 'sibb': 9, 'si': 11, 'rekk': 4}, 'portugues': {'labb': 7, 'ress': 4, 'sol': 7, 'sib': 10, 'rebb': 0, 'doss': 2, 'miss': 6, 'mibb': 2, 'sis': 12, 'la': 9, 'res': 3, 'mib': 3, 're': 2, 'fabb': 3, 'reb': 1, 'mis': 5, 'solss': 9, 'solbb': 5, 'do': 0, 'fas': 6, 'solb': 6, 'lab': 8, 'fa': 5, 'fab': 4, 'sols': 8, 'siss': 13, 'las': 10, 'lass': 11, 'dobb': -2, 'dob': -1, 'mi': 4, 'sibb': 9, 'si': 11, 'fass': 7, 'dos': 1}, 'nederlands': {'as': 8, 'ais': 10, 'ceses': -2, 'beses': 9, 'fisis': 7, 'es': 3, 'deses': 0, 'eses': 2, 'cisis': 2, 'geses': 5, 'fes': 4, 'aisis': 11, 'gisis': 9, 'ges': 6, 'fis': 6, 'ases': 7, 'c': 0, 'eeses': 2, 'feses': 3, 'ces': -1, 'e': 4, 'bis': 12, 'a': 9, 'disis': 4, 'aes': 8, 'b': 11, 'aeses': 7, 'd': 2, 'g': 7, 'f': 5, 'eisis': 6, 'des': 1, 'eis': 5, 'ees': 3, 'bisis': 13, 'cis': 1, 'gis': 8, 'bes': 10, 'dis': 3}, 'english': {'fsharp': 6, 'aflatflat': 7, 'gsharp': 8, 'gs': 8, 'dx': 4, 'fflatflat': 3, 'cff': -2, 'csharp': 1, 'cf': -1, 'fsharpsharp': 7, 'csharpsharp': 2, 'gf': 6, 'asharpsharp': 11, 'bflatflat': 9, 'ex': 6, 'esharpsharp': 6, 'cflat': -1, 'cs': 1, 'fff': 3, 'gff': 5, 'bflat': 10, 'cflatflat': -2, 'eflatflat': 2, 'bff': 9, 'cx': 2, 'aflat': 8, 'aff': 7, 'gflat': 6, 'bsharp': 12, 'asharp': 10, 'dsharp': 3, 'df': 1, 'es': 5, 'dff': 0, 'css': 2, 'dflatflat': 0, 'ass': 11, 'fflat': 4, 'ef': 3, 'fx': 7, 'c': 0, 'bss': 13, 'esharp': 5, 'dflat': 1, 'ff': 4, 'bs': 12, 'ess': 6, 'gx': 9, 'bx': 13, 'ds': 3, 'a': 9, 'dss': 4, 'gss': 9, 'b': 11, 'e': 4, 'd': 2, 'g': 7, 'f': 5, 'af': 8, 'as': 10, 'fs': 6, 'dsharpsharp': 4, 'ax': 11, 'fss': 7, 'gsharpsharp': 9, 'eff': 2, 'gflatflat': 5, 'bf': 10, 'bsharpsharp': 13, 'eflat': 3}}


def get_language(string, default = 'english'):
    if not hasattr(get_language, 'regex'):
        get_language.regex = re.compile(r'\\include +"([a-z]+)\.ly"')
    matches = get_language.regex.finditer(string)
    
    for match in matches:
        lang = match.group(1)
        if lang in language_notes:
            return lang

    return default

def get_named_key(string):
    if not hasattr(get_named_key, 'regex'):
        get_named_key.regex = re.compile(r'\\key +([a-z]+) *\\(major|minor)')
    matches = get_named_key.regex.finditer(string)

    key = None
    for match in matches:
        name = match.group(1)
        mode = match.group(2)
        if key is not None and key != (name, mode):
            return None
        key = (name, mode)

    return key

class AmbiguousTransposeException(Exception):
    pass

def get_transpose(string):
    if not hasattr(get_transpose, 'regex'):
        get_transpose.regex = re.compile(r'\\transpose +([a-z]+)\'* +([a-z]+)\'*')
    matches = get_transpose.regex.finditer(string)

    fr0m = to = prev_from = prev_to = None
    for match in matches:
        fr0m = match.group(1)
        to = match.group(2)
        if (prev_from is not None and prev_from != fr0m) \
                or (prev_to is not None and prev_to != to):
            raise AmbiguousTransposeException()
        prev_from = fr0m
        prev_to = to

    if fr0m != to:
        return (fr0m, to)
    return None

def get_key(filename):
    with open(filename) as f:
        string = f.read()

    language = get_language(string)

    k = get_named_key(string)
    if not k:
        return None
    key_name, mode = k

    try:
        root = language_notes[language][key_name]
    except Exception:
        return None

    try:
        transpose = get_transpose(string)
    except AmbiguousTransposeException:
        return None

    if transpose is not None:
        fr0m, to = transpose
        try:
            fr0m = language_notes[language][fr0m]
            to = language_notes[language][to]
        except Exception:
            return None
        root = (root + (to - fr0m)) % 12

    if mode == 'major':
        k = key.MajorKey(root)
    else:
        k = key.MinorKey(root)

    return k

if __name__ == '__main__':
    if os.path.isdir(sys.argv[1]):
        for f in glob(sys.argv[1] + '/*.ly'):
            k = get_key(f)
            if k:
                print f, k
    else:
        k = get_key(sys.argv[1])
        print k
