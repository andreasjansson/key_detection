import hashlib
import pickle
import tempfile

class Cache(object):

    read_enabled = True
    write_enabled = True
    global_prefix = ''

    def __init__(self, prefix, key = ''):
        self.name = '%s/cache_%s%s_%s.pkl' % (tempfile.gettempdir(), Cache.global_prefix, prefix, hashlib.md5(key).hexdigest())
        self._data = None

    def exists(self):
        if not Cache.read_enabled:
            False

        return self.get() is not None

    # handles broken pickle
    def get(self):
        if not Cache.read_enabled:
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
        if not Cache.write_enabled:
            return

        with open(self.name, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def set_caching_read(enabled):
        Cache.read_enabled = enabled

    @staticmethod
    def set_caching_write(enabled):
        Cache.write_enabled = enabled

    @staticmethod
    def set_caching_prefix(prefix):
        Cache.global_prefix = prefix
