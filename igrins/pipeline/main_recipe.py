from .argh_helper import arg
from .driver import get_obsset, get_obsset_from_context  # , apply_steps
from .steps import apply_steps

from ..igrins_libs.logger import info


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
                obsdate, config_file, bands, groups,
                basename_postfix=""):

    from ..igrins_libs.igrins_config import IGRINSConfig
    config = IGRINSConfig(config_file)

    fn = config.get_value('RECIPE_LOG_PATH', obsdate)

    from ..igrins_libs.recipes import RecipeLog
    recipes = RecipeLog(fn)

    selected = get_selected(recipes, replace_name_fnmatch,
                            groups)

    for band in bands:
        info("= Entering band:{}".format(band))
        for s in selected:
            # obsids = s[0]
            # frametypes = s[1]

            recipe_name = s[0].strip()
            obsids = s[1]
            frametypes = s[2]
            aux_infos = s[3]
            groupname = aux_infos["group1"]

            obsset = get_obsset(obsdate, recipe_name, band,
                                obsids, frametypes,
                                groupname=groupname, recipe_entry=aux_infos,
                                config_file=config,
                                basename_postfix=basename_postfix)
            yield obsset


driver_args = [arg("-b", "--bands", default="HK"),
               arg("-g", "--groups", default=None),
               arg("-c", "--config-file", default=None),
               arg("-v", "--verbose", default=0),
               arg("--resume-from-context-file", default=None),
               arg("--save-context-on-exception", default=False),
               arg("-d", "--debug", default=False)]


def driver_func(steps, recipe_name_fnmatch, obsdate,
                bands="HK", groups=None,
                config_file=None, debug=False, verbose=None,
                resume_from_context_file=None, save_context_on_exception=False,
                **kwargs):

    if resume_from_context_file is not None:
        import pickle
        p = pickle.load(open(resume_from_context_file, "rb"))
        resource_context = p["resource_context"]
        context_id = p["context_id"]
        obsset_desc = p["obsset_desc"]

        obsset = get_obsset_from_context(obsset_desc, resource_context)
        apply_steps(obsset, steps,
                    nskip=context_id, kwargs=kwargs)
        return

    obsset_list = [obsset for obsset in iter_obsset(recipe_name_fnmatch,
                                                    obsdate, config_file,
                                                    bands, groups)]

    for obsset in obsset_list:
        context_id = apply_steps(obsset, steps, nskip=0, kwargs=kwargs)

        if context_id is not None:  # if an exception is raised
            obsset_desc = obsset.get_descriptions()
            print("execution failed during step {context_id} of {obsset_desc}"
                  .format(context_id=context_id + 1, obsset_desc=obsset_desc))
            if save_context_on_exception:
                obsset.rs.context_stack.garbage_collect()
                p = dict(obsdate=obsdate,
                         resource_context=obsset.rs,
                         obsset_desc=obsset_desc,
                         context_id=context_id
                )

                import pickle
                pickle.dump(p, open("obsset_context.pickle", "wb"))
            return
        else:
            pass
        # print("No error")

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
