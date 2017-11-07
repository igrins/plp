from __future__ import print_function

import os


def _from_arguments(args):
    return None


def _from_config(config):
    return None


def _from_current():
    return "./recipes"


def _import_from_path(module_name, path,
                      submodule_search_locations=None):
    try:
        # Python 3.5+
        import importlib.util
        spec = importlib.util.spec_from_file_location(module_name,
                                                      path)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)

    except ImportError:
        # python 2
        import imp

        foo = imp.load_source(module_name, path)

    return foo


def load_recipes(recipe_name):

    import sys

    recipe_full_name = "recipe_" + recipe_name

    recipe_dir = _from_current()
    recipe_path0 = os.path.join(recipe_dir,
                                "__init__.py")
    recipe_path = os.path.join(recipe_dir,
                               recipe_full_name + ".py")

    if os.path.exists(recipe_path):
        print("setting")
        m = _import_from_path("recipes",
                              recipe_path0)
        sys.modules["recipes"] = m
        # from recipes import recipe_test as m
        m = _import_from_path("recipes." + recipe_full_name,
                              recipe_path)
        return m.recipes

    # import from default path
    _temp = __import__("recipes." + recipe_full_name, globals(), locals(),
                       ['recipes'], 2)
    return _temp.recipes


if __name__ == "__main__":
    load_recipes("test")
    print("done")
