from ..resource_manager.file_storage import FileStorage
from ..libs.path_info import IGRINSPath


def get_storage(config, resource_spec):
    utdate = resource_spec[0]
    path_info = IGRINSPath(config, utdate)

    return FileStorage(resource_spec, path_info)
