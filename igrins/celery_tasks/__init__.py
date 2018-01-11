import os
from celery import Celery
from ..quicklook.obsset_ql import quicklook_func


default_broker = 'pyamqp://guest@localhost//'
broker = os.environ.get('CELERY_BROKER', default_broker)

# app = Celery('tasks', broker='pyamqp://guest@localhost//')
app = Celery('tasks', broker=broker, backend='rpc://')


@app.task
def do_ql_flat(triname, obsdate, obsids, frametypes):
    config_file = "recipe.{triname}.config".format(triname=triname)

    quicklook_func(obsdate, objtypes=["FLAT"] * len(obsids),
                   bands="HK",
                   frametypes=frametypes, obsids=obsids,
                   config_file=config_file)


@app.task
def do_recipe(triname, obsdate, recipe_list):
    from .recipe_func import recipe_func
    config_file = "recipe.{triname}.config".format(triname=triname)
    for (task_name, recipe_name, group1, obsids, frametypes) in recipe_list:
        recipe_func(obsdate, task_name, recipe_name,
                    frametypes=frametypes, obsids=obsids,
                    config_file=config_file)
