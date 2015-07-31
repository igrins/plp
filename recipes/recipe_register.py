from libs.recipe_base import RecipeBase

from recipe_wvlsol_v0 import process_band

class RecipeRegister(RecipeBase):

    def run_selected_bands_with_recipe(self, utdate, selected, bands,
                                       recipe_name):
        for band in bands:
            for s in selected:
                obsids = s[0]
                print obsids
                # frametypes = s[1]

                process_band(utdate, recipe_name, band, obsids, self.config)

                # process_thar_band(utdate, self.refdate, band, obsids,
                #                   self.config)


def _register_factory(cls, recipe_name):
    _recipe_register_obj = cls()
    def register_subcommand(utdate, bands="HK",
                            starting_obsids=None,
                            config_file="recipe.config"):

        _recipe_register_obj(utdate, bands,
                             starting_obsids, config_file)

    register_subcommand.__name__ = recipe_name.lower()
    return register_subcommand

_recipe_dict = {}
for recipe_name in ["ThAr", "Sky"]:
    type_name = "RecipeRegister%s" % recipe_name
    cls = type(type_name, (RecipeRegister,),
               dict(RECIPE_NAME=recipe_name.upper()))

    r = _register_factory(cls, recipe_name)
    _recipe_dict[recipe_name] = r


def get_recipe_list():
    return _recipe_dict.values()

def get_recipe_dict():
    return _recipe_dict


# class RecipeRegisterThAr(RecipeBase):
#     RECIPE_NAME = "THAR"

# class RecipeRegisterSky(RecipeBase):
#     RECIPE_NAME = "SKY"

#_recipe_register_thar = RecipeRegisterThAr()
#_recipe_register_sky = RecipeRegisterSky()



#thar = _register_factory(cls, name)
