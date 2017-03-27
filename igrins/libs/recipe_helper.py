from path_info import IGRINSPath
from products import PipelineStorage
import os
from cal_db import CalDB

class RecipeHelper:
    def __init__(self, config_name, utdate, recipe_name="",
                 ensure_dir=False):

        from igrins_config import IGRINSConfig
        if isinstance(config_name, str):
            self.config = IGRINSConfig(config_name)
        else:
            self.config = config_name

        self.utdate = utdate
        self.refdate = self.config.get("MASTER_CAL", "REFDATE")

        self._igr_path = IGRINSPath(self.config, utdate,
                                    ensure_dir=ensure_dir)

        self._igr_storage = PipelineStorage(self._igr_path)

        self.recipe_name = recipe_name

    def get_caldb(self):
        caldb = CalDB(self, self.utdate)
        return caldb

    def get_filenames(self, band, obsids):
        return self._igr_path.get_filenames(band, obsids)

    def get_filename(self, band, obsid):
        return self._igr_path.get_filename(band, obsid)

    def get_basename(self, band, master_obsid):
        filenames = self.get_filenames(band, [master_obsid])
        basename = os.path.splitext(os.path.basename(filenames[0]))[0]
        return basename

    def get_basename_with_groupname(self, band, groupname):
        if isinstance(groupname, int):
            groupname = str(groupname)
        return self._igr_path.get_basename(band, groupname)

    def get_base_info(self, band, obsids):
        filenames = self.get_filenames(band, obsids)
        basename = os.path.splitext(os.path.basename(filenames[0]))[0]
        master_obsid = obsids[0]

        return filenames, basename, master_obsid

    def load_ref_data(self, band, spec):
        from master_calib import load_ref_data
        s = load_ref_data(self.config, band, spec)
        return s

    def get_item_path(self, item_desc, basename,
                      prevent_split=False, subdir=None):
        return self._igr_storage.get_item_path(item_desc, basename,
                                               prevent_split=prevent_split,
                                               subdir=subdir)

    def load(self, product_descs, mastername, prevent_split=False):
        return self._igr_storage.load(product_descs, mastername,
                                      prevent_split=prevent_split)

    def load_item_from_path(self, item_path):
        return self._igr_storage.load_item_from_path(item_path)

    def load_item(self, item_desc, basename):
        return self._igr_storage.load_item(item_desc, basename)

    def load1(self, product_desc, mastername,
              return_hdu_list=False, prevent_split=False):
        return self._igr_storage.load1(product_desc, mastername,
                                       return_hdu_list=return_hdu_list,
                                       prevent_split=prevent_split)

    def get_masterhdu(self, mastername):
        return self._igr_storage.get_masterhdu(mastername)

    def store_item(self, item_desc, basename, pipeline_image,
                   basename_postfix=None):
        return self._igr_storage.store_item(item_desc, basename,
                                            pipeline_image,
                                            basename_postfix=basename_postfix)

    def store(self, products, mastername, masterhdu=None, cache=True,
              basename_postfix=None):
        return self._igr_storage.store(products, mastername, 
                                       masterhdu=masterhdu, cache=cache,
                                       basename_postfix=basename_postfix)
