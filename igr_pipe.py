import sys

import igrins.igrins_libs.logger as logger

from igrins.pipeline.argh_helper import argh

from igrins.pipeline.main_recipe import driver_func, driver_args
from igrins.pipeline.steps import create_argh_command_from_steps
from igrins.igrins_recipes.recipe_prepare_recipe_logs \
    import prepare_recipe_logs

from igrins.quicklook.obsset_ql import create_argh_command_quicklook

# from igrins.pipeline.sample_steps import get_pipeline_steps

import re
p_extract = re.compile(r'(\w+)-(ab|onoff)')


from igrins.igrins_recipes import get_pipeline_steps

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
               create_argh_command("extract-arc", ["ARC_*"]),
               create_argh_command("a0v-ab", ["A0V_AB"]),
               create_argh_command("a0v-onoff", ["A0V_ONOFF"]),
               create_argh_command("stellar-ab", ["STELLAR_AB"]),
               create_argh_command("stellar-onoff", ["STELLAR_ONOFF"]),
               create_argh_command("extended-ab", ["EXTENDED_AB"]),
               create_argh_command("extended-onoff", ["EXTENDED_ONOFF"]),
               create_argh_command("plot-spec", ["A0V_*",
                                                 "STELLAR_*",
                                                 "EXTENDED_*"]),
               create_argh_command_quicklook(),
]

parser = argh.ArghParser()
parser.add_commands(recipe_list)

import igrins
print(igrins)

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
