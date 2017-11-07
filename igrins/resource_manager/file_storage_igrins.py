from ..resource_manager.file_storage import FileStorage
from ..libs.path_info import IGRINSPath


def get_storage(config, resource_spec):
    utdate, band = resource_spec
    path_info = IGRINSPath(config, utdate, band)

    return FileStorage(resource_spec, path_info)
