import os

### NEW

from .igrins_libs.resource_helper_igrins import ResourceHelper

from .igrins_libs.storage_descriptions import load_descriptions

from .igrins_libs.logger import set_level as set_log_level


DESCS = load_descriptions()


###

# def get_caldb(config_name, utdate, ensure_dir=False):
#     from .libs.recipe_helper import RecipeHelper
#     helper = RecipeHelper(config_name, utdate, ensure_dir=ensure_dir)
#     caldb = helper.get_caldb()
#     return caldb

# def get_obsset(caldb, band, recipe_name, obsids, frametypes):
#     from .libs.obs_set import ObsSet
#     obsset = ObsSet(caldb, band, recipe_name, obsids, frametypes)
#     return obsset

# def get_obsset_from_log(caldb, band, log_entry):
#     recipe_name, obsids, frametypes = (log_entry["recipe"],
#                                        log_entry["obsids"],
#                                        log_entry["frametypes"])
                                       
#     obsset = get_obsset(caldb, band, recipe_name, obsids, frametypes)
#     return obsset

# def get_recipe_log(caldb):
#     config = caldb.get_config()
#     fn0 = config.get_value('RECIPE_LOG_PATH', caldb.utdate)
#     fn = os.path.join(config.root_dir, fn0)
#     from igrins.libs.recipes import RecipeLog
#     recipe_log = RecipeLog(fn)

#     return recipe_log

# def get_obsset_from_log(caldb, band, log_entry):
#     recipe_name, obsids, frametypes = (log_entry["recipe"],
#                                        log_entry["obsids"],
#                                        log_entry["frametypes"])
                                       
#     obsset = get_obsset(caldb, band, recipe_name, obsids, frametypes)
#     return obsset


def load_config(config_file=None):
    from .igrins_libs.igrins_config import IGRINSConfig
    config = IGRINSConfig(config_file=config_file)
    return config


def load_recipe_log(obsdate, config_file=None):
    from .igrins_libs.igrins_config import IGRINSConfig
    if isinstance(config_file, IGRINSConfig):
        config = config_file
    else:
        config = IGRINSConfig(config_file=config_file)
    fn = os.path.join(os.path.dirname(config.config_file),
                      config.get_value('RECIPE_LOG_PATH', obsdate))

    from .igrins_libs.recipes import RecipeLog
    recipe_log = RecipeLog(obsdate, fn)

    return recipe_log


def get_obsset(obsdate, band, recipe_name_or_entry,
               obsids=None, frametypes=None,
               groupname=None, recipe_entry=None,
               config_file=None):
    if isinstance(recipe_name_or_entry, str):
        recipe_name = recipe_name_or_entry
    else:
        r = recipe_name_or_entry
        recipe_name = r["recipe"]
        obsids = r["obsids"]
        frametypes = r["frametypes"]
        groupname = r["group1"]

        if recipe_entry is None:
            recipe_entry = r

    from .pipeline.driver import get_obsset as _get_obsset
    obsset = _get_obsset(obsdate, recipe_name, band, obsids, frametypes,
                         groupname, recipe_entry,
                         config_file=config_file)
    return obsset


def get_obsset_helper(obsset):
    return ResourceHelper(obsset)


class CalDB(object):
    """
    This is a helper class to access the calibration files, but when no
    target obsid is necessary. Mostly for the test purpose.
    """
    def __init__(self, obsdate, band, resource_manager):
        self.obsdate = obsdate
        self.band = band
        self.rm = resource_manager

    def get_section_defs(self):
        return self.rm.storage.get_section_defs()

    def get_db_List(self):
        return ['flat_off', 'flat_on', 'register', 'wvlsol', 'distortion']

    def get_all_products_for_db(self, db_name):

        db = self.rm.resource_db._get_db(db_name)
        obsid_list, basename_list = db.get_obsid_list()
        obsid = obsid_list[0]

        d = [k for k, v in DESCS.items() if self.rm.locate(obsid, v)]

        return d

    def load(self, db_name, desc, n=0):
        if isinstance(desc, str):
            desc = DESCS[desc]

        db = self.rm.resource_db._get_db(db_name)
        obsid_list, basename_list = db.get_obsid_list()
        obsid = obsid_list[0]

        return self.rm.load(obsid, desc)


# basename = 'SDCK_20170215_0061'
# obsid = 61
# postfix=""
# section, tmpl = DESCS["ORDERS_JSON"]
# fn = tmpl.format(basename=basename, postfix=postfix)
# k = caldb.rm.context_stack.locate(section, fn)

# k = caldb.rm.locate('SDCK_20170215_0061', DESCS['VAR2D_FITS'])
# k = caldb.rm.locate(obsid, DESCS['VAR2D_FITS'])

def get_caldb(obsdate, band, config_file=None, saved_context_name=None):
    """
    obsdate, band = "20170215", "K"
    config_file = "recipe.config"

    caldb = get_caldb(obsdate, band, config_file)

    _ = caldb.get_all_products_for_db("flat_on")
    print(_)

    v = caldb.load("flat_on", "FLATON_JSON")

    """
    from .igrins_libs.igrins_config import IGRINSConfig
    from .igrins_libs.resource_manager import get_igrins_resource_manager

    if isinstance(config_file, IGRINSConfig):
        config = config_file
    else:
        config = IGRINSConfig(config_file)

    if saved_context_name is not None:
        import cPickle as pickle
        resource_manager = pickle.load(open(saved_context_name, "rb"))
    else:
        resource_manager = get_igrins_resource_manager(config, (obsdate, band))

    caldb = CalDB(obsdate, band, resource_manager)

    return caldb

