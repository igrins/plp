import os

from .compress_helper import candidate_generators


class StorageBase(object):
    def search_candidate(self, section, fn):
        fn_search_list = []
        for gen_candidate, helper in candidate_generators.values():
            fn1 = gen_candidate(fn)
            if self.exists(section, fn1):
                return fn1, helper.decompress
            fn_search_list.append(fn1)
        else:
            raise FileNotFoundError("No candidate files are found : %s" 
                                    % fn_search_list)

    def new_sectioned_storage(self, section):
        return SectionedStorage(self, section)

    def get_section_defs(self):
        raise RuntimeError("Need to be implmented by derived classes")

    def get_section(self, section):
        return self.get_section_location(section)

    def get_section_location(self, section):
        d = self.get_section_defs()

        return d[section]


class SectionedStorage(object):
    def __init__(self, storage, section):
        self.storage = storage
        self.section = section

    def search_candidate(self, fn):
        return self.sroage.search_candidate(section, fn)

    def exists(self, fn):
        return self.storage.exists(self.section, fn)

    def load(self, fn, item_type=None, check_candidate=None):
        return self.storage.load(self.section, fn,
                                 item_type=item_type,
                                 check_candidate=check_candidate)

    def store(self, fn, d, item_type=None):
        self.storage.store(self.section, fn, d,
                           item_type=item_type)

