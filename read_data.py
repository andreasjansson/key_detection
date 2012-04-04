from util import *
import sqlite3 as sqlite
import sys

def read_data(track_id, mp3, key_lab_file, writer,
              window_size = 8192, chroma_bins = 5 * 12,
              spectral_bands = 5, samp_rate = 44100):

    audio = Mp3Reader().read(mp3)
    spectrogram = generate_spectrogram(audio, window_size)
    chromagrams = []
    keylab = KeyLab(key_lab_file)

    i = 0
    for (t, spectrum) in spectrogram:

        for b in range(spectral_bands):
            start = b * window_size
            end = b * (window_size + 1)
            band = spectrum[start:end]
            chromagram = Chromagram(
                band, start, chroma_bins, samp_rate)
            chromagrams += chromagram.values

        key = keylab.key_at(t / samp_rate)
        row = [track_id, i, key] + chromagrams
        write_row(row)

        i += 1


class SqliteDataWriter:

    def __init__(self, database, chroma_bins = 5 * 12, spectral_bands = 5):
        self.chroma_bins = chroma_bins
        self.spectral_bands = spectral_bands

        try:
            self.con = sqlite.connect(database)
            self.cur = self.con.cursor()
        except sqlite.Error, e:
            print "Error %s:" % e.args[0]
            sys.exit(1)

    def create_table(self):
        chroma_columns = ["chroma_%d_%d FLOAT NOT NULL" % (i, j)
                          for i in range(self.spectral_bands)
                          for j in range(self.chroma_bins)]
        chroma_columns = ", ".join(chroma_columns)
        try:
            self.cur.execute("CREATE TABLE data(id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, track_id INT NOT NULL, i INT NOT NULL, key INT NOT NULL, %s)" % (chroma_columns))
        except sqlite.Error, e:
            print "Error %s:" % e.args[0]
            sys.exit(1)

    def write_row(self, row):
        expected_row_length = 1 + 1 + 1 + self.chroma_bins * self.spectral_bands
        if len(row) != expected_row_length:
            message = "Row length %d doesn't match expected row length %d" % \
                (len(row), expected_row_length)
            raise Exception(message)

        row = ", ".join(str(x) for x in row)
        try:
            self.cur.execute("INSERT INTO data VALUES (NULL, %s)" % row)
        except sqlite.Error, e:
            message = "Sqlite insert error %s:" % e.args[0]
            raise Exception(message)
