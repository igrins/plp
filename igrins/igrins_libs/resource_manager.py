import re

from ..resource_manager import (ResourceStack, ResourceStackWithBasename,
                                ResourceDBSet)

from .file_storage_igrins import get_storage

from .resource_db_igrins import get_igrins_db_factory

from .storage_descriptions import load_resource_def
from .item_convert import ItemConverter

from .master_calib import (query_ref_value,
                           query_ref_value_from_section,
                           query_ref_data_path,
                           get_ref_loader)


class IGRINSRefLoader(object):
    def __init__(self, config, band):
        self.config = config

        self.band = band

    def query_value(self, kind):
        kind += "_{}".format(self.band)
        return query_ref_value(self.config, band=self.band, kind=kind)

    def query_value_from_section(self, section, kind, default=None):
        kind += "_{}".format(self.band)
        return query_ref_value_from_section(self.config, self.band,
                                            section, kind, default=default)

    def query(self, kind):

        return query_ref_data_path(self.config, band=self.band, kind=kind)

    def load(self, kind, get_path=False):

        fn = query_ref_data_path(self.config, self.band, kind)
        loader = get_ref_loader(fn)
        d = loader(fn)

        if get_path:
            return fn, d
        else:
            return d

    # def fetch(self, kind):
    #     from .master_calib import fetch_ref_data
    #     fn, d = fetch_ref_data(self.config, band=self.band,
    #                            kind=kind)
    #     return fn, d



class IgrinsBasenameHelper():
    #p = re.compile(r"SDC(\w)_(\d+\w*)_(\d+)([^_]*)")
    p = re.compile(r"N(\d+\w*)S(\d+\w*)_(\w+)([^_]*)")
    p_obsid = re.compile(r"(\d+)(.*)")

    def __init__(self, obsdate, band):
        self.obsdate = obsdate
        self.band = band

    def to_basename(self, obsid):
        if isinstance(obsid, int):
            group_postfix = ""
        else:
            obsid_, group_postfix = self.p_obsid.match(obsid).groups()
            obsid = int(obsid_)

        #return "SDC{band}_{obsdate}_{obsid:04d}{group_postfix}".format(obsdate=self.obsdate,
        return "N{obsdate}S{obsid:04d}_{band}{group_postfix}".format(obsdate=self.obsdate,
                                                                       band=self.band,
                                                                       obsid=obsid,
                                                                       group_postfix=group_postfix)

    def from_basename(self, basename):
        m = self.p.match(basename)
        return str(int(m.group(3))) + m.group(4)

    def parse_basename(self, basename):
        return self.from_basename(basename)


def get_file_storage(config, resource_spec, check_candidate=False):
    return get_storage(config, resource_spec, check_candidate=check_candidate)


def get_resource_db(config, resource_spec):

    db_factory = get_igrins_db_factory(config, resource_spec)

    resource_def = load_resource_def()

    return ResourceDBSet(resource_spec,
                         db_factory, resource_def)


def get_resource_manager(config, resource_spec,
                         basename_helper=None, item_converter_class="auto",
                         check_candidate=False):

    obsdate, band = resource_spec

    base_storage = get_file_storage(config, resource_spec,
                                    check_candidate=check_candidate)

    if item_converter_class is None:
        storage = base_storage
    elif item_converter_class == "auto":
        storage = ItemConverter(base_storage)
    else:
        storage = item_converter_class(base_storage)

    resource_db = get_resource_db(config, resource_spec)

    master_ref_loader = IGRINSRefLoader(config, band)

    if basename_helper is None:
        resource_manager = ResourceStack(resource_spec, storage,
                                         resource_db,
                                         master_ref_loader=master_ref_loader)
    else:
        resource_manager = ResourceStackWithBasename(resource_spec, storage,
                                                     resource_db,
                                                     basename_helper,
                                                     master_ref_loader=master_ref_loader)

    return resource_manager


def get_igrins_resource_manager(config, resource_spec):
    obsdate, band = resource_spec

    basename_helper = IgrinsBasenameHelper(obsdate, band)

    rs = get_resource_manager(config, resource_spec,
                              basename_helper=basename_helper,
                              check_candidate=True)

    return rs
