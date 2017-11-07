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
            raise RuntimeError("No candidate files are found : %s" 
                               % fn_search_list)

