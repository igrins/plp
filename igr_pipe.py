import sys

import igrins.igrins_libs.logger as logger

from igrins.pipeline.argh_helper import argh

from igrins.pipeline.main_recipe import driver_func, driver_args
from igrins.pipeline.steps import create_argh_command_from_steps

# from igrins.pipeline.sample_steps import get_pipeline_steps

import re
p_extract = re.compile(r'(\w+)-(ab|onoff)')


def get_pipeline_steps(recipe_name):
    import igrins.igrins_recipes.recipe_flat as recipe_flat
    import igrins.igrins_recipes.recipe_register as recipe_register
    import igrins.igrins_recipes.recipe_wvlsol as recipe_wvlsol_sky
    import igrins.igrins_recipes.recipe_extract_sky as recipe_extract_sky
    import igrins.igrins_recipes.recipe_a0v_onoff as recipe_a0v_onoff
    import igrins.igrins_recipes.recipe_a0v_ab as recipe_a0v_ab
    import igrins.igrins_recipes.recipe_stellar_onoff as recipe_stellar_onoff
    import igrins.igrins_recipes.recipe_stellar_ab as recipe_stellar_ab
    import igrins.igrins_recipes.recipe_extended_onoff as recipe_extended_onoff
    import igrins.igrins_recipes.recipe_extended_ab as recipe_extended_ab

    # m = p_extract.match(recipe_name)
    # if m:
    #     recipe_name = m.group(1)

    steps = {"flat": recipe_flat.steps,
             "register-sky": recipe_register.steps,
             "wvlsol-sky": recipe_wvlsol_sky.steps,
             "extract-sky": recipe_extract_sky.steps,
             "extended-ab": recipe_extended_ab.steps,
             "extended-onoff": recipe_extended_onoff.steps,
             "stellar-ab": recipe_stellar_ab.steps,
             "stellar-onoff": recipe_stellar_onoff.steps,
             "a0v-ab": recipe_a0v_ab.steps,
             "a0v-onoff": recipe_a0v_onoff.steps
    }

    return steps[recipe_name]


def create_argh_command(recipe_name, recipe_name_fnmatch=None):
    steps = get_pipeline_steps(recipe_name)

    f = create_argh_command_from_steps(recipe_name, steps,
                                       driver_func, driver_args,
                                       recipe_name_fnmatch=recipe_name_fnmatch)
    return f


recipe_list = [create_argh_command("flat"),
               create_argh_command("register-sky", ["SKY", "SKY_AB"]),
               create_argh_command("wvlsol-sky", ["SKY", "SKY_AB"]),
               create_argh_command("extract-sky", ["SKY", "SKY_AB"]),
               create_argh_command("a0v-ab", ["A0V_AB"]),
               create_argh_command("a0v-onoff", ["A0V_ONOFF"]),
               create_argh_command("stellar-ab", ["STELLAR_AB"]),
               create_argh_command("stellar-onoff", ["STELLAR_ONOFF"]),
               create_argh_command("extended-ab", ["EXTEND_AB"]),
               create_argh_command("extended-onoff", ["EXTEND_ONOFF"]),
]

parser = argh.ArghParser()
parser.add_commands(recipe_list)

# for k, v in subcommands.items():
#     parser.add_commands(v, namespace=k)

if __name__ == '__main__':
    import numpy
    numpy.seterr(all="ignore")
    argv = sys.argv[1:]
    if "--debug" in argv:
        argv.remove("--debug")
        logger.set_level("debug")

    argh.dispatch(parser, argv=argv)
