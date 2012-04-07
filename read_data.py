from util import *
import sqlite3 as sqlite
import sys
import os
from glob import glob
import argparse
import os.path as path

class ChromaReader:

    def __init__(self, window_size = 8192, chroma_bins = 3 * 12,
                 spectral_bands = [(33, 262), (262, 2093), (2093, 5000)]):
        self.window_size = window_size
        self.chroma_bins = chroma_bins
        self.spectral_bands = spectral_bands

    def read_data(self, track_id, mp3, key_lab_file, writer):
        samp_rate, audio = Mp3Reader().read(mp3)
        spectrogram = generate_spectrogram(audio, self.window_size)
        keylab = KeyLab(key_lab_file)
        
        i = 0
        for (t, spectrum) in spectrogram:
            chromagrams = []
            for band in self.spectral_bands:
                chromagram = Chromagram(spectrum, samp_rate, self.chroma_bins, band)
                chromagrams += chromagram.values.tolist()
        
            key = keylab.key_at(t / samp_rate)
            if key is None:
                key = -1
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

class SqliteDataReader:
    pass

def insert_data(mp3_root, lab_root, reader, writer):
    track_id = 0
    mp3_folders = set(os.listdir(mp3_root))
    lab_folders = set(os.listdir(lab_root))
    shared_folders = mp3_folders.intersection(lab_folders)

    for folder in shared_folders:
        mp3_folder = mp3_root + "/" + folder
        lab_folder = lab_root + "/" + folder
        mp3_files = set(map(lambda s: path.basename(s).replace(".mp3", ""),
                            glob(mp3_folder + "/*.mp3")))
        lab_files = set(map(lambda s: path.basename(s).replace(".lab", ""),
                            glob(lab_folder + "/*.lab")))

        shared_files = mp3_files.intersection(lab_files)

        for file in shared_files:
            mp3_file = mp3_folder + "/" + file + ".mp3"
            lab_file = lab_folder + "/" + file + ".lab"
            print("Inserting (%s, %s) as track_id %d\n" % (mp3_file, lab_file, track_id))
            reader.read_data(track_id, mp3_file, lab_file, writer)
            track_id += 1
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Insert chroma data.')
    parser.add_argument('--mp3', default='/home/andreas/music/The Beatles')
    parser.add_argument('--lab', default='/home/andreas/data/beatles_annotations/keylab/the_beatles')
    parser.add_argument('--db', default='data.db')
    args = parser.parse_args()
    reader = ChromaReader()
    writer = SqliteDataWriter(args.db)
    writer.create_table()
    insert_data(args.mp3, args.lab, reader, writer)
