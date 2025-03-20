import os
from six.moves import configparser as ConfigParser
from six.moves import cStringIO as StringIO

from ..igrins_libs.logger import warning

default_config_content = """[DEFAULT]
MASTER_CAL_DIR=master_calib
INDATA_PATH=indata/%(UTDATE)s
OUTDATA_PATH=outdata/%(UTDATE)s
PRIMARY_CALIB_PATH=calib/primary/%(UTDATE)s
SECONDARY_CALIB_PATH=calib/secondary/%(UTDATE)s
QA_PATH=%(OUTDATA_PATH)s/qa
HTML_PATH=html/%(UTDATE)s
RECIPE_LOG_PATH=recipe_logs/%(UTDATE)s.recipes
"""


class IGRINSConfig(object):
    def __init__(self, config_file=None):

        if config_file is None:
            config_file = 'recipe.config'

        import os.path

        config_file = os.path.abspath(config_file)
        self.config_file = config_file

        #_ = os.path.abspath(config_file)
        self.root_dir = os.path.dirname(config_file)

        self.config = ConfigParser.ConfigParser()

        fp = StringIO(default_config_content)
        self.config.read_file(fp)

        read_file = self.config.read(config_file)
        if not read_file:
            warning("no {} is found. Internal default will be used."
                 .format(config_file))

        self.master_cal_dir = os.path.join(self.root_dir,
                                           self.config.get("DEFAULT",
                                                           "MASTER_CAL_DIR"))

        import os
        self.config.read(os.path.join(self.master_cal_dir,
                                      "master_cal.config"))

    def get_value(self, option, utdate):
        return self.config.get("DEFAULT", option,
                               raw=0, vars=dict(UTDATE=utdate))

    def get(self, section, kind, **kwargs):
        return self.config.get(section, kind, raw=0, vars=kwargs)


def get_config(config):
    if isinstance(config, IGRINSConfig):
        config = config
    else:
        config = IGRINSConfig(config)

    return config


if __name__ == "__main__":

    config = IGRINSConfig()
    s = config.get_value('RECIPE_LOG_PATH', "20140525")
    print(s, type(s))

    # print config.get("DEFAULT", 'INDATA_PATH', 0, dict(UTDATE="20140525"))

    # print config.get("DEFAULT", 'QA_PATH', 0, dict(UTDATE="20140525"))
