#from products import PipelineProducts

from igrins.libs.logger import logger

def _parse_starting_obsids(starting_obsids):
    if starting_obsids is not None:
        starting_obsids = map(int, starting_obsids.split(","))
        return starting_obsids
    else:
        return None

def _parse_groups(groups):
    if groups is not None:
        groups = [s.strip() for s in groups.split(",")]
        return groups
    else:
        return None

def filter_a0v(a0v, a0v_obsid, group2):

    # print master_obsid, a0v_obsid
    if a0v is not None:
        if a0v_obsid is not None:
            raise ValueError("a0v-obsid option is not allowed "
                             "if a0v opption is used")
        elif str(a0v).upper() == "GROUP2":
            a0v = group2
    else:
        if a0v_obsid is not None:
            a0v = a0v_obsid
        else:
            # a0v, a0v_obsid is all None. Keep it as None
            pass

    return a0v

def get_selected(recipes, recipe_name, starting_obsids, groups):
    if starting_obsids is not None:
        logger.warn("'starting-obsids' option is deprecated, "
                    "please use 'groups' option.")
        if groups is not None:
            raise ValueError("'starting-obsids' option is not allowed"
                             " when 'groups' option is used.")
        else:
            starting_obsids_parsed = _parse_starting_obsids(starting_obsids)

            selected = recipes.select_fnmatch(recipe_name,
                                              starting_obsids_parsed)
    else:
        groups_parsed = _parse_groups(groups)

        selected = recipes.select_fnmatch_by_groups(recipe_name,
                                                    groups_parsed)
        logger.info("selected recipe: {}".format(selected))

    return selected


class RecipeBase(object):
    """ The derived mus define RECIPE_NAME attribute and must implement
        run_selected_bands method.

        RECIPE_NAME can be a string, or a sequence of strings which is
        interpreted as a fnmatch translater.
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def set_recipe_name(self, recipe_name):
        self.RECIPE_NAME = recipe_name

    def _validate_bands(self, bands):
        if not bands in ["H", "K", "HK"]:
            raise ValueError("bands must be one of 'H', 'K' or 'HK'")

    def get_recipe_name(self, utdate):
        fn = self.config.get_value('RECIPE_LOG_PATH', utdate)
        return fn

    def get_recipes(self, utdate):
        fn = self.get_recipe_name(utdate)
        from recipes import Recipes #load_recipe_list, make_recipe_dict
        return Recipes(fn)

    def run_selected_bands_with_recipe(self, utdate, selected, bands):
        # just ignore recipe
        selected2 = [_[1:] for _ in selected]
        self.run_selected_bands(utdate, selected2, bands)

    def __call__(self, utdate, bands="HK",
                 starting_obsids=None, groups=None,
                 config_file="recipe.config"):
        self.process(utdate, bands, starting_obsids, groups,
                     config_file=config_file)

    def process(self, utdate, bands="HK",
                starting_obsids=None, 
                groups=None,
                config_file="recipe.config",
                **kwargs):

        from igrins_config import IGRINSConfig
        self.config = IGRINSConfig(config_file)

        self.refdate = self.config.get("MASTER_CAL", "REFDATE")

        self._validate_bands(bands)

        recipes = self.get_recipes(utdate)

        selected = get_selected(recipes, self.RECIPE_NAME,
                                starting_obsids, groups)

        self.run_selected_bands_with_recipe(utdate, selected, bands,
                                            **kwargs)
