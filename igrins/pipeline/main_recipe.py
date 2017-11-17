from .argh_helper import arg
from .driver import get_obsset  # , apply_steps
from .steps import apply_steps


# for testing purposes
def _parse_groups(groups):
    if groups is None:
        return None
    else:
        return [s.strip() for s in groups.split(",")]


def get_selected(recipes, recipe_name, groups):
    groups_parsed = _parse_groups(groups)

    selected = recipes.select_fnmatch_by_groups(recipe_name,
                                                groups_parsed)
    # logger.info("selected recipe: {}".format(selected))

    return selected


def iter_obsset(replace_name_fnmatch,
                obsdate, config_file, bands, groups):

    from ..libs.igrins_config import IGRINSConfig
    config = IGRINSConfig(config_file)

    fn = config.get_value('RECIPE_LOG_PATH', obsdate)

    from ..libs.recipes import Recipes2
    recipes = Recipes2(fn)

    selected = get_selected(recipes, replace_name_fnmatch,
                            groups)

    for band in bands:
        print("entering band:{}".format(band))
        for s in selected:
            print(s)
            # obsids = s[0]
            # frametypes = s[1]

            recipe_name = s[0].strip()
            obsids = s[1]
            frametypes = s[2]
            aux_infos = s[3]
            groupname = aux_infos["GROUP1"]

            obsset = get_obsset(obsdate, recipe_name, band,
                                obsids, frametypes, config,
                                groupname=groupname, recipe_entry=aux_infos)
            yield obsset


driver_args = [arg("-b", "--bands", default="HK"),
               arg("-g", "--groups", default=None),
               arg("-c", "--config-file", default=None),
               arg("-v", "--verbose", default=None),
               arg("-d", "--debug", default=False)]


def driver_func(steps, recipe_name_fnmatch, obsdate,
                bands="HK", groupname=None,
                config_file=None, debug=False, verbose=None,
                **kwargs):

    for obsset in iter_obsset(recipe_name_fnmatch, obsdate,
                              config_file, bands, groupname):
        apply_steps(obsset, steps, nskip=0, kwargs=kwargs)
        # print(obsset)




# def execute_recipe(obsdate, recipe_name, **kwargs):
#     config_name = "recipe.config.igrins128"

#     # obsdate = "20150120"
#     obsids = [45, 46]

#     frametypes = "AB"

#     recipe_name = "STELLAR_AB"

#     band = "H"

#     context_name = "test_context_extract.pickle"

#     if True:  # rerun from saved
#         nskip = 4
#         save_context_name = None
#         saved_context_name = context_name
#     else:
#         nskip = 0
#         save_context_name = context_name  # context_name
#         saved_context_name = None

#     obsset = get_obsset(obsdate, recipe_name, band,
#                         obsids, frametypes, config_name,
#                         saved_context_name=saved_context_name)

#     apply_steps(obsset, extract.steps[:],
#                 nskip=nskip)

#     if save_context_name is not None:
#         obsset.rs.save_pickle(open(save_context_name, "wb"))
