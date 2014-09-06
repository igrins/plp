class RecipeBase(object):
    """ The derived mus define RECIPE_NAME attribute and must implement
        run_selected_bands method.
    """

    def _validate_bands(self, bands):
        if not bands in ["H", "K", "HK"]:
            raise ValueError("bands must be one of 'H', 'K' or 'HK'")

    def get_recipe_name(self, utdate):
        return "%s.recipes" % utdate

    def get_recipes(self, utdate):
        fn = self.get_recipe_name(utdate)
        from libs.recipes import Recipes #load_recipe_list, make_recipe_dict
        return Recipes(fn)

    def parse_starting_obsids(self, starting_obsids):
        if starting_obsids is not None:
            starting_obsids = map(int, starting_obsids.split(","))
            return starting_obsids
        else:
            return None

    def __call__(self, utdate, bands="HK",
                 starting_obsids=None, config_file="recipe.config"):

        from libs.igrins_config import IGRINSConfig
        self.config = IGRINSConfig(config_file)

        self.refdate = self.config.get_value('REFDATE', utdate)

        self._validate_bands(bands)

        recipes = self.get_recipes(utdate)

        starting_obsids_parsed = self.parse_starting_obsids(starting_obsids)

        selected = recipes.select(self.RECIPE_NAME, starting_obsids_parsed)

        self.run_selected_bands(utdate, selected, bands)
