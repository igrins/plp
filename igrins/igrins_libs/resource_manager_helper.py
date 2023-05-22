import os
from igrins import DESCS
from igrins.igrins_libs.resource_manager import get_igrins_resource_manager

class ResourceManagerWorkaround:
    def __init__(self, config, obsdate, default_desc=DESCS["RAWIMAGE"]):
        self.obsdate = obsdate
        self.config = config
        self._d = dict()
        self._desc = default_desc

    def get_rm(self, obsdate, band):
        k = (obsdate, band)
        if k not in self._d:
            v = get_igrins_resource_manager(self.config, (obsdate, band))
            self._d[k] = v
        return self._d[k]

    def get_path(self, obsdate, band, obsid):
        # we first search for directory in self.obsdate with obsdate (which can
        # be different)
        rm_this = self.get_rm(self.obsdate, band)
        rm_other = self.get_rm(obsdate, band)
        section, fn0 = rm_other.get_section_n_fn(obsid, self._desc)

        # p = rm_this.storage.storage.locate(section, fn0)
        p = rm_this.storage.storage._get_path(section, fn0)
        return p

    def locate(self, obsdate, band, obsid):
        # we first search for directory in self.obsdate with obsdate (which can
        # be different)
        rm_this = self.get_rm(self.obsdate, band)
        rm_other = self.get_rm(obsdate, band)
        section, fn0 = rm_other.get_section_n_fn(obsid, self._desc)

        # p = rm_this.storage.storage.locate(section, fn0)
        try:
            p = rm_this.storage.storage.locate(section, fn0)
        except RuntimeError:
            p = None

        if p is None:
            p = rm_other.storage.storage.locate(section, fn0)

        return p


def test():
    from igrins.igrins_libs.igrins_config import IGRINSConfig
    config_file="recipe.config"
    config = IGRINSConfig(config_file)

    obsdate = "20160303"
    obsdate1, obsid = "20160303", "0001"
    band = "H"
    rmw = ResourceManagerWorkaround(config, obsdate)
    # rm_old = rmw.get_rm(obsdate1, band)
    # old_fn = rmw.get_path(obsdate1, band, obsid)
    old_fn = rmw.locate(obsdate1, band, obsid)
    print(old_fn)

    # print("old", old_fn)
    # rm_new = rmd.get(obsdate, obsid)
    # new_fn = rm_new.locate(new_obsid, DESCS["RAWIMAGE"])
