from libs.recipe_base import RecipeBase

class RecipeFactoryProcessBand(RecipeBase):
    """
    The subclass musth define 'process_band' attribute, whose call signature is
     def process_band(utdate, recipe_name, band, obsids, config_name):
         pass
    """
    def run_selected_bands_with_recipe(self, utdate, selected, bands):
        for band in bands:
            for s in selected:
                recipe_name = s[0].strip()
                obsids = s[1]

                self.process_band(utdate, recipe_name, band, obsids,
                                  self.config)

def new_recipe_class(type_name, recipe_name, process_band_func):
    cls = type(type_name, (RecipeFactoryProcessBand,),
               dict(RECIPE_NAME=recipe_name,
                    process_band=staticmethod(process_band_func)))
    return cls

def new_recipe_func(function_name, recipe_cls):

    def _recipe_func(utdate, bands="HK",
                     starting_obsids=None,
                     config_file="recipe.config"):

        _recipe_obj = recipe_cls()
        _recipe_obj.process(utdate, bands,
                            starting_obsids, config_file)

    _recipe_func.__name__ = function_name.lower()
    return _recipe_func
