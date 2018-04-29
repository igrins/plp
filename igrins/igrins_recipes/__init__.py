def get_path():
    from .. import igrins_recipes
    return igrins_recipes.__path__

translation_map = {"wvlsol-sky": "wvlsol",
                   "register-sky": "register"}

def get_pipeline_steps(recipe_name,
                       parent_module_name="igrins.igrins_recipes"):
    # This may not work for python3
    import imp
    cur_path = get_path()
    recipe_name = translation_map.get(recipe_name, recipe_name)
    mod_name = "recipe_{}".format(recipe_name.replace("-", "_"))

    fp, path, desc = imp.find_module(mod_name, cur_path)

    try:
        mod = imp.load_module(".".join([parent_module_name, mod_name]),
                              fp, path, desc)
        return mod.steps

    finally:
        if fp:
            fp.close()

