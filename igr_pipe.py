try:
    import argh
except ImportError:
    import sys
    sys.path.append("./external/argh")

    import argh

import recipes.recipe_flat
import recipes.recipe_thar
import recipes.recipe_wvlsol_sky
#from recipes.recipe_distort_sky import distortion_sky
from recipes.recipe_extract import (a0v_ab, stellar_ab,
                                    a0v_onoff, stellar_onoff,
                                    extended_ab, extended_onoff)
from recipes.recipe_extract_plot import plot_spec
from recipes.recipe_publish_html import publish_html

from recipes.recipe_prepare_recipe_logs import prepare_recipe_logs

recipe_list = [recipes.recipe_flat.flat,
               recipes.recipe_thar.thar,
               recipes.recipe_wvlsol_sky.sky_wvlsol,
               #distortion_sky,
               a0v_ab,
               stellar_ab,
               a0v_onoff,
               stellar_onoff,
               extended_ab,
               extended_onoff,
               plot_spec,
               publish_html,
               prepare_recipe_logs
               ]

parser = argh.ArghParser()
parser.add_commands(recipe_list)

if __name__ == '__main__':
    argh.dispatch(parser)
