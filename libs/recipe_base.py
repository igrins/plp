import os
from libs.path_info import IGRINSPath
#from libs.products import PipelineProducts
from libs.products import ProductDB, PipelineStorage

class RecipeBase(object):
    """ The derived mus define RECIPE_NAME attribute and must implement
        run_selected_bands method.
    """

    def _validate_bands(self, bands):
        if not bands in ["H", "K", "HK"]:
            raise ValueError("bands must be one of 'H', 'K' or 'HK'")

    def get_recipe_name(self, utdate):
        fn = self.config.get_value('RECIPE_LOG_PATH', utdate)
        return fn

    def get_recipes(self, utdate):
        fn = self.get_recipe_name(utdate)
        from libs.recipes import Recipes #load_recipe_list, make_recipe_dict
        return Recipes(fn)

    def parse_starting_obsids(self, starting_obsids):
        if starting_obsids is not None:
            starting_obsids = map(int, starting_obsids.split(","))
            return starting_obsids
        else:
            return None

    def __call__(self, utdate, bands="HK",
                 starting_obsids=None, config_file="recipe.config"):

        from libs.igrins_config import IGRINSConfig
        self.config = IGRINSConfig(config_file)

        self.refdate = self.config.get_value('REFDATE', utdate)

        self._validate_bands(bands)

        recipes = self.get_recipes(utdate)

        starting_obsids_parsed = self.parse_starting_obsids(starting_obsids)

        selected = recipes.select(self.RECIPE_NAME, starting_obsids_parsed)

        self.run_selected_bands(utdate, selected, bands)



class ProcessBase(object):
    def __init__(self, utdate, refdate, config):
        """
        cr_rejection_thresh : pixels that deviate significantly from the profile are excluded.
        """
        self.utdate = utdate
        self.refdate = refdate
        self.config = config

        self.igr_path = IGRINSPath(config, utdate)

        self.igr_storage = PipelineStorage(self.igr_path)


    def prepare(self, band, obsids, frametypes, load_a0v_db=True):

        igr_path = self.igr_path

        self.obj_filenames = igr_path.get_filenames(band, obsids)

        self.master_obsid = obsids[0]

        self.tgt_basename = os.path.splitext(os.path.basename(self.obj_filenames[0]))[0]

        self.db = {}
        self.basenames = {}


        db_types_calib = ["flat_off", "flat_on", "thar", "sky"]

        for db_type in db_types_calib:

            db_name = igr_path.get_section_filename_base("PRIMARY_CALIB_PATH",
                                                         "%s.db" % db_type,
                                                         )
            self.db[db_type] = ProductDB(db_name)


        # db on output path
        db_types = ["a0v"]

        for db_type in db_types:

            db_name = igr_path.get_section_filename_base("OUTDATA_PATH",
                                                         "%s.db" % db_type,
                                                         )
            self.db[db_type] = ProductDB(db_name)

        # to get basenames
        db_types = ["flat_off", "flat_on", "thar", "sky"]
        if load_a0v_db:
            db_types.append("a0v")

        for db_type in db_types:
            self.basenames[db_type] = self.db[db_type].query(band,
                                                             self.master_obsid)

def get_pr(utdate, config_file="recipe.config"):
    from libs.igrins_config import IGRINSConfig
    #from jj_recipe_base import ProcessBase
    config = IGRINSConfig(config_file)
    refdate = config.get_value("REFDATE", None)
    pr = ProcessBase(utdate, refdate, config)

    return pr
