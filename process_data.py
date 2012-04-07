from util import *

class Processor:

    def __init__(self, database = 'data.db', keymap = simple_keymap):
        self.db = RawDb(database)
        self.keymap = simple_keymap

    # TODO: unit test
    def get_markov_matrix(self):
        rows = self.db.select(['track_id', 'i', 'key'], order = 'track_id, i')
        key_count = len(self.keymap.keys()) 
        basic_markov = [0] * key_count
        prev_track = None
        prev_key = -1
        for row in rows:
            track = row['track_id']
            i = row['i']
            key = row['key']
            if prev_track is not None and \
                    key > -1 and prev_key > -1:
                basic_markov[(prev_key - key) % key_count] += 1
            prev_track = track
            prev_key = key
        markov = [[0] * key_count for _ in self.keymap]
        for i in range(key_count):
            for j in range(key_count):
                value = basic_markov[(i - j) % key_count]
                markov[i][j] = value
            colsum = float(sum(markov[i]))
            if colsum > 0:
                for j in range(key_count):
                    markov[i][j] /= colsum
        return markov

    def get_chromagrams():
        pass
