from util import *
import sqlite3 as sqlite
import sys

def read_data(track_id, mp3, key_lab_file, writer,
              window_size = 8192, chroma_bins = 3 * 12,
              spectral_bands = [(33, 262), (262, 2093), (2093, 5000)]):

    samp_rate, audio = Mp3Reader().read(mp3)
    spectrogram = generate_spectrogram(audio, window_size)
    keylab = KeyLab(key_lab_file)

    i = 0
    for (t, spectrum) in spectrogram:

        chromagrams = []
        for band in spectral_bands:
            chromagram = Chromagram(spectrum, samp_rate, chroma_bins, band)
            chromagrams += chromagram.values.tolist()

        key = keylab.key_at(t / samp_rate)
        row = [track_id, i, key] + chromagrams
        writer.write_row(row)

        i += 1


class SqliteDataWriter:

    def __init__(self, database, chroma_bins = 3 * 12, bands_count = 3):
        self.chroma_bins = chroma_bins
        self.bands_count = bands_count

        try:
            self.con = sqlite.connect(database)
            self.cur = self.con.cursor()
        except sqlite.Error, e:
            print "Error %s:" % e.args[0]
            sys.exit(1)

    def create_table(self):
        chroma_columns = ["chroma_%d_%d FLOAT NOT NULL" % (i, j)
                          for i in range(self.bands_count)
                          for j in range(self.chroma_bins)]
        chroma_columns = ", ".join(chroma_columns)
        try:
            self.cur.execute("CREATE TABLE raw(id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, track_id INT NOT NULL, i INT NOT NULL, key INT NOT NULL, %s)" % (chroma_columns))
            self.con.commit()
        except sqlite.Error, e:
            print "Error %s:" % e.args[0]
            sys.exit(1)

    def write_row(self, row):
        expected_row_length = 1 + 1 + 1 + self.chroma_bins * self.bands_count
        if len(row) != expected_row_length:
            message = "Row length %d doesn't match expected row length %d" % \
                (len(row), expected_row_length)
            raise Exception(message)

        row = ", ".join(str(x) for x in row)
        try:
            self.cur.execute("INSERT INTO raw VALUES (NULL, %s)" % row)
            self.con.commit()
        except sqlite.Error, e:
            message = "Sqlite insert error %s:" % e.args[0]
            raise Exception(message)
