import numpy as np

def load_recipe_list(fn):
    dtype=[('OBJNAME', 'S128'), ('OBJTYPE', 'S128'), ('GROUP1', 'i'), ('GROUP2', 'i'), ('EXPTIME', 'f'), ('RECIPE', 'S128'), ('OBSIDS', 'S1024')]
    d = np.genfromtxt(fn, delimiter=",", names=True, comments="#",
                      dtype=dtype)
    recipe_list = []
    for row in d:
        recipe_name = row["RECIPE"].strip()
        obsids  = map(int, row["OBSIDS"].strip().split())
        recipe_list.append((recipe_name, obsids, row))

    return recipe_list

def make_recipe_dict(recipe_list):
    recipe_dict = {}
    for recipe_name, obsids, row in recipe_list:
        recipe_dict.setdefault(recipe_name, []).append((obsids, row))
    return recipe_dict
