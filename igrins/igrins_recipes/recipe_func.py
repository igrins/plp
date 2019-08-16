from . import get_pipeline_steps
from ..pipeline.steps import apply_steps
from ..pipeline.driver import get_obsset

from ..pipeline.main_recipe import driver_func_obsset


# _recipe_func : bare minimum wrapper around apply steps
# recipe_func : wrapper around driver_func_obsset

def _recipe_func(obsdate, task_name, recipe_name,
                 band,
                 obsids=None, frametypes=None,
                 groupname=None,
                 **kwargs):

    # print(obsdate, obsids)
    config_file = kwargs.pop("config_file", None)
    if config_file is None:
        config_file = "recipe.config"

    steps = get_pipeline_steps(task_name)

    obsset = get_obsset(obsdate, recipe_name, band,
                        obsids=obsids, frametypes=frametypes,
                        groupname=None,
                        config_file=config_file)

    apply_steps(obsset, steps, kwargs=kwargs)


def recipe_func(task_name,
                obsdate, band,
                obsids=None, frametypes=None,
                groupname=None,
                recipe_name=None,
                config_file=None,
                log_level="INFO", debug=False,
                **kwargs):

    # print(obsdate, obsids)
    # config_file = kwargs.pop("config_file", None)
    if config_file is None:
        config_file = "recipe.config"

    if recipe_name is None:
        recipe_name = task_name

    runner_config = dict(log_level=log_level, debug=debug)

    steps = get_pipeline_steps(task_name)

    obsset = get_obsset(obsdate, recipe_name, band,
                        obsids=obsids, frametypes=frametypes,
                        groupname=None,
                        config_file=config_file, runner_config=runner_config)

    # apply_steps(obsset, steps, kwargs=kwargs)
    if task_name is None:
        task_name = recipe_name

    driver_func_obsset(task_name, obsdate, steps, [obsset], **kwargs)


if True:
    import inspect

    def merge_signature(f1, f2):
        s1 = inspect.signature(f1)
        s2 = inspect.signature(f2)

        filtered_ss = (list(s1.parameters.values())[:-1]
                       + [_s for _s in s2.parameters.values()
                          if ((_s.default is not inspect._empty) or
                              (_s.kind is inspect.Parameter.VAR_KEYWORD))])

        return inspect.Signature(filtered_ss)

    recipe_func.__signature__ = merge_signature(recipe_func,
                                                driver_func_obsset)
