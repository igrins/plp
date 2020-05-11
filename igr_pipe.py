#! /usr/bin/env python

import sys

import igrins.igrins_libs.logger as logger
from igrins.igrins_recipes.arghed_recipes import get_recipe_list
from igrins.pipeline.argh_helper import argh

if __name__ == '__main__':
    parser = argh.ArghParser()
    recipe_list = get_recipe_list()
    parser.add_commands(recipe_list)

    import numpy
    numpy.seterr(all="ignore")
    argv = sys.argv[1:]
    if "--debug" in argv:
        argv.remove("--debug")
        logger.set_level("debug")

    argh.dispatch(parser, argv=argv)
