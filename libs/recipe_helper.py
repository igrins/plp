from path_info import IGRINSPath
from products import PipelineStorage
import os
from cal_db import CalDB

class RecipeHelper:
    def __init__(self, config_name, utdate, recipe_name=""):

        from igrins_config import IGRINSConfig
        if isinstance(config_name, str):
            self.config = IGRINSConfig(config_name)
        else:
            self.config = config_name

        self.utdate = utdate
        self.refdate = self.config.get("MASTER_CAL", "REFDATE")

        self.igr_path = IGRINSPath(self.config, utdate)

        self.igr_storage = PipelineStorage(self.igr_path)

        self.recipe_name = recipe_name

    def get_caldb(self):
        caldb = CalDB(self, self.utdate)
        return caldb

    def get_filenames(self, band, obsids):
        return self.igr_path.get_filenames(band, obsids)

    def get_basename(self, band, master_obsid):
        filenames = self.get_filenames(band, [master_obsid])
        basename = os.path.splitext(os.path.basename(filenames[0]))[0]
        return basename

    def get_base_info(self, band, obsids):
        filenames = self.get_filenames(band, obsids)
        basename = os.path.splitext(os.path.basename(filenames[0]))[0]
        master_obsid = obsids[0]

        return filenames, basename, master_obsid

    def load_ref_data(self, band, spec):
        from master_calib import load_ref_data
        s = load_ref_data(self.config, band, spec)
        return s
