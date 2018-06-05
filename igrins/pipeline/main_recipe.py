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


def get_selected(recipes, recipe_name_fnmatch, groups):
    groups_parsed = _parse_groups(groups)

    selected = recipes.select_fnmatch_by_groups(recipe_name_fnmatch,
                                                groups_parsed)
    # logger.info("selected recipe: {}".format(selected))

    return selected


def iter_obsset(recipe_name_fnmatch,
                obsdate, config_file, bands, groups,
                basename_postfix=""):

    from ..igrins_libs.igrins_config import IGRINSConfig
    config = IGRINSConfig(config_file)

    fn = config.get_value('RECIPE_LOG_PATH', obsdate)

    from ..igrins_libs.recipes import RecipeLog
    recipes = RecipeLog(obsdate, fn)

    selected = get_selected(recipes, recipe_name_fnmatch,
                            groups)

    for band in bands:
        # info("= Entering band:{}".format(band))
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


def _save_context(obsdate, obsset, context_id, outname):
    info("Saving context to '{}'".format(outname))
    obsset.rs.context_stack.garbage_collect()
    obsset_desc = obsset.get_descriptions()
    p = dict(obsdate=obsdate,
             resource_context=obsset.rs,
             obsset_desc=obsset_desc,
             context_id=context_id
    )

    import pickle
    pickle.dump(p, open(outname, "wb"))


def _load_context(outname):
    import pickle
    p = pickle.load(open(outname, "rb"))
    info("Loading context from '{}'".format(outname))
    resource_context = p["resource_context"]
    context_id = p["context_id"]
    obsset_desc = p["obsset_desc"]
    # print(obsset_desc)
    # if step_range is None:
    #     step_slice = slice(context_id, None)

    obsset = get_obsset_from_context(obsset_desc, resource_context)
    return obsset


driver_args = [arg("-b", "--bands", default="HK", choices=["HK", "H", "K"]),
               arg("-g", "--groups", default=None),
               arg("-c", "--config-file", default=None),
               arg("-v", "--verbose", default=0),
               arg("--override-recipe-name", default=False),
               arg("--step-range", default=None),
               arg("--context-name", default="context_{obsdate}_{recipe_name}_{groupname}{basename_postfix}_{context_id}.pickle"),
               arg("--save-context-if", default="never",
                   choices=["never", "exception", "always"]),
               arg("-d", "--debug", default=False)]


def driver_func(steps, recipe_name_fnmatch, obsdate,
                bands="HK", groups=None,
                config_file=None, debug=False, verbose=None,
                override_recipe_name=False,
                # resume_from_context_file=None,
                save_context_if="never",
                context_name="context_{obsdate}_{recipe_name}_{groupname}{basename_postfix}_{context_id}.pickle",
                # save_context_on_exception=False,
                step_range=None,
                **kwargs):

    # FIXME : should check if 'resume_from_context_file' and 'step_range' are
    # not set together.

    if step_range is not None:
        _se = [k for k in step_range.split(":")]
        if len(_se)  == 1:
            k = int(_se[0])
            _s, _e = k, k + 1
        elif len(_se) == 2:
            _s, _e= [int(k) if k.strip() else None for k in _se]
        else:
            raise ValueError("incorrect step_range: {}".format(ste_range))

        step_slice = slice(_s, _e)
    else:
        step_slice = slice(None, None)

    if override_recipe_name:
        if groups is None:
            raise RuntimeError("override_recipe_name should specify groups.")

        recipe_name_fnmatch = ["*"]

    obsset_list = [obsset for obsset in iter_obsset(recipe_name_fnmatch,
                                                    obsdate, config_file,
                                                    bands, groups)]

    if save_context_if == "always":
        save_context = True
        save_context_on_exception = True
    elif save_context_if == "exception":
        save_context = False
        save_context_on_exception = True
    elif save_context_if == "never":
        save_context = False
        save_context_on_exception = False
    else:
        raise ValueError("unknown save_context_if argument: {}".
                         format(save_context_if))

    if save_context_on_exception:
        def on_raise(obsset, context_id):
            outname = context_name.format(obsdate=obsdate,
                                          context_id=context_id,
                                          **obsset_desc)
            _save_context(obsdate, obsset, context_id, outname)
    else:
        def on_raise(obsset, context_id):
            obsset_desc = obsset.get_descriptions()
            print("execution failed during step {context_id} of {obsset_desc}"
                  .format(context_id=context_id + 1, obsset_desc=obsset_desc))

    for obsset in obsset_list:
        obsset_desc = obsset.get_descriptions()

        if step_slice.start:

            context_id=step_slice.start
            outname = context_name.format(obsdate=obsdate,
                                          context_id=context_id,
                                          **obsset_desc)
            obsset = _load_context(outname)

        apply_steps(obsset, steps, step_slice=step_slice,
                    kwargs=kwargs,
                    on_raise=on_raise)

        if (save_context or
            (step_slice.stop is not None and step_slice.stop < len(steps))):
            context_id=step_slice.stop

            outname = context_name.format(obsdate=obsdate,
                                          context_id=context_id,
                                          **obsset_desc)
            _save_context(obsdate, obsset, context_id, outname)
