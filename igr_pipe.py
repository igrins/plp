from igrins.recipes.argh_helper import argh

import igrins.recipes.recipe_flat
#import igrins.recipes.recipe_thar
#import igrins.recipes.recipe_wvlsol_sky

from igrins.recipes.recipe_wvlsol_sky2 import wvlsol_sky, sky_wvlsol

#from igrins.recipes.recipe_distort_sky import distortion_sky
from igrins.recipes.recipe_extract import (a0v_ab, stellar_ab,
                                    a0v_onoff, stellar_onoff,
                                    extended_ab, extended_onoff)
from igrins.recipes.recipe_extract_plot import plot_spec
from igrins.recipes.recipe_publish_html import publish_html

from igrins.recipes.recipe_prepare_recipe_logs import prepare_recipe_logs
from igrins.recipes.recipe_tell_wvsol import tell_wvsol, wvlsol_tell
from igrins.recipes.recipe_make_sky import make_sky

recipe_list = [igrins.recipes.recipe_flat.flat,
               #igrins.recipes.recipe_thar.thar,
               #igrins.recipes.recipe_wvlsol_sky.sky_wvlsol,
               #igrins.recipes.recipe_wvlsol_sky.wvlsol_sky,
               wvlsol_sky,
               sky_wvlsol,
               #distortion_sky,
               a0v_ab,
               stellar_ab,
               a0v_onoff,
               stellar_onoff,
               extended_ab,
               extended_onoff,
               plot_spec,
               publish_html,
               prepare_recipe_logs,
               tell_wvsol,
               wvlsol_tell,
               make_sky
               ]

import igrins.recipes.recipe_register as recipe_register
recipe_list.extend(recipe_register.get_command_list())

from igrins.recipes.recipe_divide_a0v import divide_a0v
recipe_list.extend([divide_a0v])

parser = argh.ArghParser()
parser.add_commands(recipe_list)

# for k, v in subcommands.items():
#     parser.add_commands(v, namespace=k)

if __name__ == '__main__':
    import numpy
    numpy.seterr(all="ignore")
    import sys
    argv = sys.argv[1:]
    if "--debug" in argv:
        argv.remove("--debug")
        # print("--debug")
        from igrins.libs.logger import set_level
        set_level("debug")

    argh.dispatch(parser, argv=argv)
