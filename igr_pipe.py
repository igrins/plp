from argh_helper import argh

import recipes.recipe_flat
import recipes.recipe_thar
import recipes.recipe_wvlsol_sky

from recipes.recipe_wvlsol_sky2 import wvlsol_sky2

#from recipes.recipe_distort_sky import distortion_sky
from recipes.recipe_extract import (a0v_ab, stellar_ab,
                                    a0v_onoff, stellar_onoff,
                                    extended_ab, extended_onoff)
from recipes.recipe_extract_plot import plot_spec
from recipes.recipe_publish_html import publish_html

from recipes.recipe_prepare_recipe_logs import prepare_recipe_logs
from recipes.recipe_tell_wvsol import tell_wvsol, wvlsol_tell
from recipes.recipe_make_sky import make_sky

recipe_list = [recipes.recipe_flat.flat,
               #recipes.recipe_thar.thar,
               recipes.recipe_wvlsol_sky.sky_wvlsol,
               recipes.recipe_wvlsol_sky.wvlsol_sky,
               wvlsol_sky2,
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

from recipes.recipe_register import register_sky, register_thar, thar

register_sky = argh.arg('--do-ab', default=False, action='store_true')(register_sky)
recipe_list.extend([register_sky, register_thar, thar])

from recipes.recipe_divide_a0v import divide_a0v
recipe_list.extend([divide_a0v])

parser = argh.ArghParser()
parser.add_commands(recipe_list)

# for k, v in subcommands.items():
#     parser.add_commands(v, namespace=k)

if __name__ == '__main__':
    import numpy
    numpy.seterr(all="ignore")
    argh.dispatch(parser)
