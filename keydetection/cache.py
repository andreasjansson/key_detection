import hashlib
import pickle
import tempfile

class Cache(object):

    enabled = True

    def __init__(self, prefix, key):
        self.name = '%s/cache_%s_%s.pkl' % (tempfile.gettempdir(), prefix, hashlib.md5(key).hexdigest())
        self._data = None

    def exists(self):
        if not Cache.enabled:
            False

        return self.get() is not None

    # handles broken pickle
    def get(self):
        if not Cache.enabled:
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
        if not Cache.enabled:
            return

        with open(self.name, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def set_caching_enabled(enabled):
        Cache.enabled = enabled

