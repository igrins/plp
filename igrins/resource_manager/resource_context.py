
class ReadCache(object):
    def __init__(self, storage):
        self.storage = storage
        self._cache = {}

    def load(self, section, fn, item_type=None):
        if (section, fn, item_type) not in self._cache:
            d = self.storage.load(section, fn, item_type=item_type)
            self._cache[(section, fn, item_type)] = d
        else:
            d = self._cache[(section, fn, item_type)]

        return d

    def describe(self):
        for (section, fn, item_type) in self._cache.keys():
            print("read-cache")
            print(section, fn, item_type)

class ResourceContext():
    def __init__(self, name, read_cache=None):
        self.name = name
        self._cache = {}
        self.status = "open"
        self._read_cache = read_cache  # just for the debugging purpose.

        self._cache_only_items = set()

    def describe(self):
        print("context: {}".format(self.name))
        for k, (item_type, buf) in self._cache.items():
            if k not in self._cache_only_items:
                print(k)
            else:
                print(k, "cache-only")

    def close(self, e="close"):
        self.status = e

    def debug(self, msg, data):
        pass

    def iter_cache(self):
        return self._cache.items()

    def store(self, section, fn, buf, item_type=None, cache_only=False):
        k = (section, fn)
        self._cache[k] = (item_type, buf)
        self.debug("store", dict(section=section, fn=fn,
                                 item_type=item_type, buf=buf))

        if cache_only:
            self._cache_only_items.add(k)
        else:
            self._cache_only_items.discard(k)

    def load(self, section, fn, item_type=None):

        self.debug("load", dict(section=section, fn=fn,
                                item_type=item_type))

        item_type, buf = self._cache[(section, fn)]
        return buf


class ResourceContextStack():
    def __init__(self, storage):
        self.storage = storage

        self.current = None
        self.context_list = []
        self._reset_read_cache()

    def _reset_read_cache(self):
        self.default_read_cache = ReadCache(self.storage)
        self.read_cache = self.default_read_cache

    def garbage_collect(self):
        self._reset_read_cache()

        for context in self.context_list:
            kill_list = []
            for k, (item_type, buf) in context.iter_cache():
                if k not in context._cache_only_items:
                    kill_list.append(k)
            for k in kill_list:
                print("kill {}/{}".format(context.name, k))
                del context._cache[k]

    def new_context(self, context_name, reset_read_cache=False):
        self.current = ResourceContext(context_name)
                                       # read_cache=self.read_cache)
        self.context_list.append(self.current)

        if reset_read_cache or (self.read_cache is None):
            self.read_cache = ReadCache(self.storage)

    def abort_context(self, context_name,
                      pop_last_context=True, reset_read_cache=True):
        indx = -1
        context = self.context_list[indx]
        assert context.name == context_name

        if pop_last_context:
            self.context_list.pop(indx)

        if reset_read_cache:
            self._reset_read_cache()

        self.current.close()
        self.current = None

    def close_context(self):
        try:
            for k, (item_type, buf) in self.current.iter_cache():
                if k not in self.current._cache_only_items:
                    section, fn = k
                    self.storage.store(section, fn, buf,
                                       item_type=item_type)
        except Exception as e:
            self.current.close(e)
            raise
        else:
            self.current.close()

        self.current = None
        # self.read_cache = self.default_read_cache

    # @property
    # def current(self):
    #     if self.context_list:
    #         return self.context_list[-1]
    #     else:
    #         return None

    def store(self, section, fn, buf, item_type=None,
              cache_only=False):
        if self.current is not None:
            self.current.store(section, fn, buf, item_type=item_type,
                               cache_only=cache_only)

        else:
            self.storage.store(section, fn, buf,
                               item_type=item_type)

    def load(self, section, fn, item_type=None):
        # check the write cache in the context_stack
        for context in self.context_list[::-1]:
            try:
                return context.load(section, fn, item_type=item_type)
            except KeyError:
                pass

        # then check the read cache.
        if self.current is not None:
            return self.read_cache.load(section, fn, item_type)
        else:
            return self.storage.load(section, fn, item_type)

