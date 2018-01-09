import os
from celery import Celery
from ..quicklook.obsset_ql import quicklook_func


default_broker = 'pyamqp://guest@localhost//'
broker = os.environ.get('CELERY_BROKER', default_broker)

# app = Celery('tasks', broker='pyamqp://guest@localhost//')
app = Celery('tasks', broker=broker, backend='rpc://')


@app.task
def do_flat(triname, obsdate, obsids, frametypes):
    config_file = "recipe.{triname}.config".format(triname=triname)
    quicklook_func(obsdate, objtypes="FLAT",
                   bands="HK",
                   frametypes=frametypes, obsids=obsids,
                   config_file=config_file)
