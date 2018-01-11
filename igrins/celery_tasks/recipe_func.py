from __future__ import print_function

from ..pipeline.driver import get_obsset

from ..igrins_recipes import get_pipeline_steps
from ..pipeline.steps import apply_steps


def recipe_func(obsdate, task_name, recipe_name,
                obsids=None, frametypes=None,
                bands="HK", **kwargs):

    print(obsdate, obsids)
    config_file = kwargs.pop("config_file", None)
    if config_file is not None:
        config_file = "recipe.config"

    steps = get_pipeline_steps(task_name)

    print(task_name, steps)

    for b in bands:
        obsset = get_obsset(obsdate, recipe_name, b,
                            obsids=obsids, frametypes=frametypes,
                            config_file=config_file)

        apply_steps(obsset, steps, nskip=0, kwargs=kwargs)
