from ..resource_manager import (ResourceManager,
                                ResourceDBSet)

from ..resource_manager.file_storage_igrins import get_storage
from ..resource_manager.resource_db_igrins \
    import get_igrins_db_factory

from .storage_descriptions import load_resource_def


def get_file_storage(config, resource_spec):
    return get_storage(config, resource_spec)


def get_resource_db(config, resource_spec):

    db_factory = get_igrins_db_factory(config, resource_spec)

    resource_def = load_resource_def()

    return ResourceDBSet(resource_spec,
                         db_factory, resource_def)


def get_resource_manager(config, resource_spec):

    storage = get_file_storage(config, resource_spec)

    resource_db = get_resource_db(config, resource_spec)

    resource_manager = ResourceManager(resource_spec, storage,
                                       resource_db)

    return resource_manager
