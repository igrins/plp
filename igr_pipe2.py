import sys

from igrins.recipes.argh_helper import argh
import igrins.libs.logger as logger

from igrins.pipeline.main_recipe import driver_func, driver_args
from igrins.pipeline.steps import create_argh_command_from_steps

# from igrins.pipeline.sample_steps import get_pipeline_steps


def get_pipeline_steps(recipe_name):
    import igrins.recipes.recipe_flat2 as recipe_flat
    import igrins.recipes.recipe_register2 as recipe_register
    steps = {"flat": recipe_flat.steps,
             "register-sky": recipe_register.steps}
    return steps[recipe_name]


def create_argh_command(recipe_name, recipe_name_fnmatch=None):
    steps = get_pipeline_steps(recipe_name)

    f = create_argh_command_from_steps(recipe_name, steps,
                                       driver_func, driver_args,
                                       recipe_name_fnmatch=recipe_name_fnmatch)
    return f


flat = create_argh_command("flat")
register_sky = create_argh_command("register-sky", ["SKY", "SKY_AB"])
recipe_list = [flat, register_sky]

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
