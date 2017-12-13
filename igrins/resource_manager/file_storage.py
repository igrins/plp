import os
from .base_storage import StorageBase
from ..utils.file_utils import ensure_dir


class FileStorage(StorageBase):
    def __init__(self, resource_spec, path_info, check_candidate=False):

        self.utdate, self.band = resource_spec

        self.path_info = path_info
        self._check_candidate = check_candidate

    def _get_path(self, section, fn):

        section_dir = self.path_info.get_section_path(section)

        file_path = os.path.join(section_dir, fn)

        return file_path

    def exists(self, section, fn):
        file_path = self._get_path(section, fn)

        return os.path.exists(file_path)

    def load(self, section, fn, item_type=None, check_candidate=None):
        if check_candidate is None:
            check_candidate = self._check_candidate

        if check_candidate:
            fn, decompress = self.search_candidate(section, fn)
        else:
            decompress = None

        file_path = self._get_path(section, fn)
        r = open(file_path).read()

        if decompress is None:
            return r
        else:
            return decompress(r)

    def store(self, section, fn, d, item_type=None):
        file_path = self._get_path(section, fn)
        ensure_dir(os.path.dirname(file_path))
        open(file_path, "wb").write(d)
