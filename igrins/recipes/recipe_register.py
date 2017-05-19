from argh_helper import argh

from process_wvlsol_v0 import process_band

from igrins.libs.recipe_factory import new_recipe_class, new_recipe_func

_recipe_class_register_thar = new_recipe_class("RecipeRegisterThAr",
                                               "THAR", process_band)

_recipe_class_register_sky = new_recipe_class("RecipeRegisterSky",
                                              ["SKY", "SKY_AB"],
                                              process_band)

_command_names = []

register_thar = new_recipe_func("register_thar",
                                _recipe_class_register_thar)

if argh.assembling.SUPPORTS_ALIASES:
    register_thar = argh.decorators.aliases('thar')(register_thar)
    _command_names.append(register_thar)
else:
    thar = new_recipe_func("thar",
                           _recipe_class_register_thar)
    _command_names.extend([register_thar, thar])

register_sky = new_recipe_func("register_sky",
                               _recipe_class_register_sky)

register_sky = argh.arg('--do-ab', default=False, action='store_true')(register_sky)

_command_names.append(register_sky)

def get_command_list():
    return _command_names

__all__ = get_command_list
