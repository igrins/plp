from collections import OrderedDict

from ..pipeline.steps import create_argh_command_from_steps
from ..pipeline.main_recipe import driver_func, driver_args
from ..pipeline.main_recipe import driver_func_obsset
from . import get_pipeline_steps
from .argh_helper import get_default_values
from .argh_helper import merge_signature, _merge_signature

from ..igrins_recipes.recipe_prepare_recipe_logs \
    import (prepare_recipe_logs, show_recipe_logs)


from ..quicklook.obsset_ql import (create_argh_command_quicklook,
                                   create_argh_command_noise_guard)

# Adding recipes
# add 'igrins_recipes/recipe_[NAME].py and define 'steps' variable in the module.


def create_argh_command(command_name, recipe_name_fnmatch=None,
                        recipe_name_exclude=None):
    steps = get_pipeline_steps(command_name)

    f = create_argh_command_from_steps(command_name, steps,
                                       driver_func, driver_args,
                                       recipe_name_fnmatch=recipe_name_fnmatch,
                                       recipe_name_exclude=recipe_name_exclude)
    return f


def get_recipe_list():
    recipe_list = [prepare_recipe_logs,
                   show_recipe_logs,
                   create_argh_command("flat"),
                   create_argh_command("combine", ["A0V_*",
                                                   "STELLAR_*",
                                                   "EXTENDED_*"]),
                   create_argh_command("combine-sky", ["SKY", "SKY_AB"]),
                   create_argh_command("register-dry", ["FLAT"]),
                   create_argh_command("register-sky", ["SKY", "SKY_AB"]),
                   create_argh_command("wvlsol-sky", ["SKY", "SKY_AB"]),
                   create_argh_command("wvlsol-dry", ["FLAT"]),
                   create_argh_command("extract-sky", ["SKY", "SKY_AB"]),
                   create_argh_command("extract-arc", ["ARC_*"]),
                   create_argh_command("a0v-ab", ["A0V_AB"]),
                   create_argh_command("a0v-onoff", ["A0V_ONOFF"]),
                   create_argh_command("stellar-ab", ["STELLAR_AB"]),
                   create_argh_command("stellar-ab-pp", ["STELLAR_AB"]),
                   create_argh_command("stellar-onoff", ["STELLAR_ONOFF"]),
                   create_argh_command("extended-ab", ["EXTENDED_AB"]),
                   create_argh_command("extended-onoff", ["EXTENDED_ONOFF"]),
                   create_argh_command("plot-spec", ["A0V_*",
                                                     "STELLAR_*",
                                                     "EXTENDED_*"]),
                   create_argh_command("divide-a0v", ["STELLAR_*",
                                                      "EXTENDED_*"]),
                   create_argh_command("analyze-dark", ["DARK"]),
                   create_argh_command("analyze-flat",
                                       ["FLAT"]),
                   create_argh_command("analyze-sky",
                                       ["SKY", "*_ONOFF", "*_AB"],
                                       recipe_name_exclude=["SKY_AB"]),
                   create_argh_command_quicklook(),
                   create_argh_command_noise_guard()
    ]

    return recipe_list


def make_recipe_func(argh_entrant):
    from inspect import signature

    def _f(obsdate, **kwargs):
        kw2 = get_default_values(argh_entrant.argh_args)
        unknown_arg_names = set(kwargs.keys()).difference(kw2.keys())
        if unknown_arg_names:
            raise RuntimeError("unknown keyword arguments: {}".
                               format(unknown_arg_names))

        argh_entrant(obsdate, **kwargs)

    _sig = _merge_signature(signature(driver_func),
                            signature(driver_func_obsset))
    _f.__signature__ = _merge_signature(signature(_f), _sig)

    return _f


def get_callable_recipes():
    recipes = OrderedDict()

    for argh_entrant in get_recipe_list():
        if hasattr(argh_entrant, "argh_name"):

            _f = make_recipe_func(argh_entrant)
            recipes[argh_entrant.argh_name] = _f

        elif callable(argh_entrant) and hasattr(argh_entrant, "__name__"):
            recipes[argh_entrant.__name__] = argh_entrant

    return recipes


if False:
    from igrins.igrins_recipes.arghed_recipes import get_callable_recipes
    recipes = get_callable_recipes()
    kk = recipes.keys()
    print(kk)
