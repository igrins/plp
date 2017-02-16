from argh_helper import argh

from process_wvlsol_v0 import process_band

from libs.recipe_factory import new_recipe_class, new_recipe_func

_recipe_class_register_thar = new_recipe_class("RecipeRegisterThAr",
                                               "THAR", process_band)

_recipe_class_register_sky = new_recipe_class("RecipeRegisterSky",
                                              ["SKY", "*_AB"],
                                              #["SKY"],
                                              process_band)

register_thar = new_recipe_func("register_thar",
                                _recipe_class_register_thar)

register_sky = new_recipe_func("register_sky",
                               _recipe_class_register_sky)

register_sky = argh.arg('--do-ab', default=False, action='store_true')(register_sky)

thar = new_recipe_func("thar",
                       _recipe_class_register_thar)

__all__ = register_thar, register_sky, thar
