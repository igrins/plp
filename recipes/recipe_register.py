from libs.recipe_base import RecipeBase

from recipe_wvlsol_v0 import process_band

class RecipeRegister(RecipeBase):

    def run_selected_bands_with_recipe(self, utdate, selected, bands):
        for band in bands:
            for s in selected:
                recipe_name = s[0].strip()
                obsids = s[1]
                #print obsids
                # frametypes = s[1]

                #print utdate, recipe_name, band, obsids, self.config
                process_band(utdate, recipe_name, band, obsids, self.config)

                # process_thar_band(utdate, self.refdate, band, obsids,
                #                   self.config)

_class_dict = {}
for recipe_name in ["ThAr", "Sky"]:
    type_name = "RecipeRegister%s" % recipe_name
    cls = type(type_name, (RecipeRegister,),
               dict(RECIPE_NAME=recipe_name.upper()))
    _class_dict[recipe_name] = cls


def _register_factory(recipe_cls, recipe_name, function_name_prefix=""):

    def _command(utdate, bands="HK",
                 starting_obsids=None,
                 config_file="recipe.config"):

        _recipe_register_obj = recipe_cls()
        _recipe_register_obj.process(utdate, bands,
                                     starting_obsids, config_file)

    _command.__name__ = (function_name_prefix+recipe_name).lower()
    return _command



def get_recipe_list(function_name_prefix=""):

    _recipe_list = []
    for recipe_name, recipe_cls in _class_dict.items():
        r = _register_factory(recipe_cls, recipe_name,
                              function_name_prefix=function_name_prefix)
        _recipe_list.append(r)

    return _recipe_list

def get_recipe_dict(function_name_prefix=""):
    l = get_recipe_list(function_name_prefix=function_name_prefix)
    return dict((f.__name__, f) for f in l)
