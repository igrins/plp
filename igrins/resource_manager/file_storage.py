import os


class FileStorage(object):
    def __init__(self, resource_spec, path_info):

        self.utdate, self.band = resource_spec

        self.path_info = path_info

    def _get_path(self, section, fn):

        section_dir = self.path_info.get_section_path(section)

        file_path = os.path.join(section_dir, fn)

        return file_path

    def load(self, section, fn, item_type=None):
        file_path = self._get_path(section, fn)
        return open(file_path).read()

    def store(self, section, fn, d, item_type=None):
        file_path = self._get_path(section, fn)
        open(file_path, "wb").write(d)
