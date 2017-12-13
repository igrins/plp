from ..resource_manager.file_storage import FileStorage
from .path_info import IGRINSPath


def get_storage(config, resource_spec, check_candidate=False):
    utdate, band = resource_spec
    path_info = IGRINSPath(config, utdate, band)

    return FileStorage(resource_spec, path_info,
                       check_candidate=check_candidate)
