import hashlib
import pickle

CACHING = True

class Cache(object):

    def __init__(self, prefix, key):
        self.name = 'cache_%s_%s.pkl' % (prefix, hashlib.md5(key).hexdigest())
        self._data = None

    def exists(self):
        if not CACHING:
            False

        return self.get() is not None

    # handles broken pickle
    def get(self):
        if not CACHING:
            return None

        if self._data is not None:
            return self._data
        try:
            with open(self.name, 'r') as f:
                self._data = pickle.load(f)
                return self._data
        except Exception:
            return None

    def set(self, data):
        if not CACHING:
            return

        with open(self.name, 'wb') as f:
            pickle.dump(data, f)

