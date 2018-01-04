import sys

import igrins.igrins_libs.logger as logger

from igrins.pipeline.argh_helper import argh

from igrins.pipeline.main_recipe import driver_func, driver_args
from igrins.pipeline.steps import create_argh_command_from_steps
from igrins.igrins_recipes.recipe_prepare_recipe_logs \
    import prepare_recipe_logs

# from igrins.pipeline.sample_steps import get_pipeline_steps

import re
p_extract = re.compile(r'(\w+)-(ab|onoff)')


def get_pipeline_steps(recipe_name):
    from igrins.igrins_recipes import (recipe_flat,
                                       recipe_register,
                                       recipe_wvlsol as recipe_wvlsol_sky,
                                       recipe_extract_sky,
                                       recipe_a0v_onoff,
                                       recipe_a0v_ab,
                                       recipe_stellar_onoff,
                                       recipe_stellar_ab,
                                       recipe_extended_onoff,
                                       recipe_extended_ab)

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


recipe_list = [prepare_recipe_logs,
               create_argh_command("flat"),
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
