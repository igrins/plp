import importlib

translation_map = {"wvlsol-sky": "wvlsol",
                   "register-sky": "register",
                   "register-thar": "register",
                   "wvlsol-thar": "wvlsol",
                   }

def get_pipeline_steps(recipe_name,
                       parent_module_name="igrins.igrins_recipes"):


    recipe_name = translation_map.get(recipe_name, recipe_name)
    mod_name = "recipe_{}".format(recipe_name.replace("-", "_"))

    mod = importlib.import_module("." + mod_name, package=parent_module_name)

    return mod.steps
