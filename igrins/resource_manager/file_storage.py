import six
import os
from .base_storage import StorageBase
from ..utils.file_utils import ensure_dir


class FileStorage(StorageBase):
    def __init__(self, resource_spec, path_info=None, check_candidate=False):

        # self.obsdate, self.band = resource_spec
        self.resource_spec = resource_spec

        self.path_info = path_info.sections

        self._check_candidate = check_candidate

    def get_section(self, section):
        if hasattr(self.path_info, "sections"):
            return self.path_info.sections.get(section)
        else:
            return self.path_info.get(section)

    def _get_path(self, section, fn):

        section_dir = self.get_section(section)

        file_path = os.path.join(section_dir, fn)

        return file_path

    def exists(self, section, fn):
        file_path = self._get_path(section, fn)

        return os.path.exists(file_path)

    def _locate(self, section, fn, item_type=None, check_candidate=None):
        if check_candidate is None:
            check_candidate = self._check_candidate

        if check_candidate:
            fn, decompress = self.search_candidate(section, fn)
        else:
            decompress = None

        file_path = self._get_path(section, fn)
        return file_path, decompress

    def locate(self, section, fn, item_type=None, check_candidate=None):
        file_path, decompress = self._locate(section, fn,
                                             item_type, check_candidate)
        return file_path

    def load(self, section, fn, item_type=None, check_candidate=None):
        file_path, decompress = self._locate(section, fn,
                                             item_type, check_candidate)

        r = open(file_path, "rb").read()

        if decompress is None:
            return r
        else:
            return decompress(r)

    def store(self, section, fn, d, item_type=None):
        file_path = self._get_path(section, fn)
        ensure_dir(os.path.dirname(file_path))

        if isinstance(d, six.binary_type):
            open(file_path, "wb").write(d)
        else:
            open(file_path, "w").write(d)

