from util import *
import argparse

class Processor:

    def __init__(self, input_db = RawTable('data.db'), keymap = simple_keymap):
        self.input_db = input_db
        self.keymap = simple_keymap

    # TODO: unit test
    def get_markov_matrix(self):
        rows = self.input_db.select(['track_id', 'key'], order = 'track_id, i')
        key_count = len(self.keymap.keys()) 
        basic_markov = [0] * key_count
        prev_track = None
        prev_key = -1
        for row in rows:
            track = row['track_id']
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

    def get_chroma_mean(self, table = TunedTable('data.db'), bands = 3,
                        keymap = simple_keymap):
        rows = table.select(['key'] + dict(table.get_chroma_columns()).keys(),
                            as_dict = False)
        totals = {}
        for row in rows:
            offset, base = get_key_base(row[0], keymap)
            values = row[1:]
            values = roll_bands(values, offset, bands)
            if base not in totals:
                totals[base] = [0] * (12 * bands)
            for i, value in enumerate(values):
                totals[base][i] += value
        for key in totals:
            key_sum = sum(totals[key])
            for i, value in enumerate(totals[key]):
                totals[key][i] /= key_sum
        totals = set_implicit_keys(totals, keymap)
        return totals

    def rows_by_track(self, column_names):
        track_id = 1
        while True:
            rows = self.input_db.select(column_names, 'track_id = ' + str(track_id), 'i',
                                        as_dict = False)
            if not len(rows):
                break
            yield rows
            track_id += 1

    def tune(self, output_writer = TunedTable('data.db'), bins_per_pitch = 3, bands = 3):
        tuner = Tuner(bins_per_pitch, bands)
        column_names = ['track_id', 'i', 'key'] + \
            dict(self.input_db.get_chroma_columns()).keys()
        for rows in self.rows_by_track(column_names):
            meta_rows = map(lambda x: x[:3], rows)
            value_rows = map(lambda x: x[3:], rows)
            tuned_rows = tuner.tune(value_rows)
            for meta_row, tuned_row in zip(meta_rows, tuned_rows):
                print("Inserting track: %d, i: %d" % (meta_row[0], meta_row[1]))
                output_writer.insert(list(meta_row) + tuned_row)

    def get_chromagrams():
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process chroma data.')
    parser.add_argument("--tune", action = "store_true")
    parser.add_argument("--db", default = "data.db")    
    args = parser.parse_args()
    if args.tune:
        processor = Processor(RawTable(args.db))
        tuned_table = TunedTable(args.db)
        try:
            tuned_table.create_table()
        except:
            tuned_table.truncate_table()
        processor.tune(TunedTable(args.db))
    else:
        parser.error("Missing parameter")
        print "hello"
