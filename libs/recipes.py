import numpy as np

def load_recipe_list(fn):
    dtype=[('OBJNAME', 'S128'), ('OBJTYPE', 'S128'), ('GROUP1', 'i'), ('GROUP2', 'i'), ('EXPTIME', 'f'), ('RECIPE', 'S128'), ('OBSIDS', 'S1024'),  ('FRAMETYPES', 'S1024')]
    d = np.genfromtxt(fn, delimiter=",", names=True, comments="#",
                      dtype=dtype)
    recipe_list = []
    for row in d:
        recipe_name = row["RECIPE"].strip()
        obsids  = map(int, row["OBSIDS"].strip().split())
        frametypes  = row["FRAMETYPES"].strip().split()
        recipe_list.append((recipe_name, obsids, frametypes, row))

    return recipe_list

def make_recipe_dict(recipe_list):
    recipe_dict = {}
    for recipe_name, obsids, frametypes, row in recipe_list:
        recipe_dict.setdefault(recipe_name, []).append((obsids, frametypes, row))
    return recipe_dict

class Recipes(object):
    def __init__(self, fn):
        self._fn = fn
        self.recipe_list = load_recipe_list(fn)
        self.recipe_dict = make_recipe_dict(self.recipe_list)

    def select(self, recipe_name, starting_obsids=None):
        if recipe_name not in self.recipe_dict:
            return []

        if starting_obsids is None:
            return self.recipe_dict[recipe_name]

        selected = []
        selected_obsids = []
        for _ in self.recipe_dict[recipe_name]:
            obsids = _[0]
            if obsids[0] in starting_obsids:
                selected.append(_)
                selected_obsids.append(obsids[0])

        if len(selected_obsids) != len(starting_obsids):
            remained_obsids = set(starting_obsids) - set(selected_obsids)
            raise RuntimeError("some obsids is not correct : %s" % \
                               ", ".join(sorted(remained_obsids)))
        else:
            return selected
