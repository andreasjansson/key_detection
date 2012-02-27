#import audio_library

class Key:
    def __init__(self, key, time):
        self.key = key
        self.time = time


class Algorithm:

    def __init__(self, mp3_file):
        self.audio = self._read_mp3(mp3_file)
        self.keys = []

    def execute(self):
        raise NotImplementedError()

    def _read_mp3:
        return [0] * 32768 * 10 # 10 seconds of nothing


