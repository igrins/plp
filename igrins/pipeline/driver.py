""" Pipeline Driver """

from ..igrins_libs.resource_manager import get_igrins_resource_manager
from ..igrins_libs.igrins_config import IGRINSConfig
from ..igrins_libs.obs_set import ObsSet


def get_obsset(obsdate, recipe_name, band,
               obsids, frametypes, config_name,
               groupname=None, recipe_entry=None, saved_context_name=None):

    # from igrins import get_obsset
    # caldb = get_caldb(config_name, obsdate, ensure_dir=True)

    if isinstance(config_name, IGRINSConfig):
        config = config_name
    else:
        config = IGRINSConfig(config_name)

    if saved_context_name is not None:
        import cPickle as pickle
        resource_manager = pickle.load(open(saved_context_name, "rb"))
    else:
        resource_manager = get_igrins_resource_manager(config, (obsdate, band))

    obsset = ObsSet(resource_manager, recipe_name, obsids, frametypes,
                    groupname=groupname, recipe_entry=recipe_entry)

    return obsset


def get_obsset_from_context(obsset_desc, resource_manager):

    recipe_name = obsset_desc["recipe_name"]
    obsids = obsset_desc["obsids"]
    frametypes = obsset_desc["frametypes"]
    groupname = obsset_desc["groupname"]

    obsset = ObsSet(resource_manager, recipe_name, obsids, frametypes,
                    groupname=groupname)  # ``, recipe_entry=recipe_entry)

    return obsset
