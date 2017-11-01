
class ReadCache(object):
    def __init__(self, storage):
        self.storage = storage
        self._cache = {}

    def load(self, section, fn, item_type=None):
        if (section, fn, item_type) not in self._cache:
            d = self.storage.load(section, fn, item_type=item_type)
            self._cache[(section, fn, item_type)] = d
        else:
            d = self._cache[(section, fn, item_type)] = d

        return d


class ResourceContext():
    def __init__(self, name, read_cache=None):
        self.name = name
        self._cache = {}
        self.status = "open"
        self._read_cache = read_cache  # just for the debugging purpose.

        self._cache_only_items = set()

    def close(self, e="close"):
        self.status = e

    def debug(self, msg, data):
        pass

    def store(self, section, fn, buf, item_type=None, cache_only=False):
        k = (section, fn, item_type)
        self._cache[k] = buf
        self.debug("store", dict(section=section, fn=fn,
                                 item_type=item_type, buf=buf))

        if cache_only:
            self._cache_only_items.add(k)
        else:
            self._cache_only_items.discard(k)

    def load(self, section, fn, item_type=None):

        self.debug("load", dict(section=section, fn=fn,
                                item_type=item_type))

        return self._cache[(section, fn, item_type)]


class ResourceContextStack():
    def __init__(self, storage):
        self.storage = storage

        self.context_list = []
        self.read_cache = None

    def new_context(self, context_name, reset_read_cache=False):
        context = ResourceContext(context_name, read_cache=self.read_cache)
        self.context_list.append(context)

        if reset_read_cache or (self.read_cache is None):
            self.read_cache = ReadCache(self.storage)

    def close_context(self):
        try:
            for k, buf in self.current._cache.items():
                if k not in self.current._cache_only_items:
                    section, fn, item_type = k
                    self.storage.store(section, fn, buf,
                                       item_type=item_type)
        except Exception as e:
            self.current.close(e)
        else:
            self.current.close()

    @property
    def current(self):
        if self.context_list:
            return self.context_list[-1]
        else:
            return None

    def store(self, section, fn, buf, item_type=None,
              cache_only=False):
        self.current.store(section, fn, buf, item_type=item_type,
                           cache_only=cache_only)

    def load(self, section, fn, item_type=None):
        # check the write cache in the context_stack
        for context in self.context_list[::-1]:
            try:
                return context.load(section, fn, item_type=item_type)
            except KeyError:
                pass

        # then check the read cache.
        return self.read_cache.load(section, fn, item_type)
