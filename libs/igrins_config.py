import ConfigParser
import warnings
import os

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

        import StringIO

        fp = StringIO.StringIO(default_config_content)
        self.config.readfp(fp)

        read_file = self.config.read(config_file)
        if not read_file:
            warnings.warn("no {} is found. Internal default will be used."
                          .format(config_file))

        self.master_cal_dir = os.path.join(self.root_dir,
                                           self.config.get("DEFAULT",
                                                           "MASTER_CAL_DIR",
                                                           0))

        import os
        self.config.read(os.path.join(self.master_cal_dir,
                                      "master_cal.config"))

    def get_value(self, option, utdate):
        return self.config.get("DEFAULT", option, 0, dict(UTDATE=utdate))

    def get(self, section, kind, **kwargs):
        return self.config.get(section, kind, 0, kwargs)


def get_config(config):
    if isinstance(config, IGRINSConfig):
        config = config
    else:
        config = IGRINSConfig(config)

    return config


if __name__ == "__main__":

    config = IGRINSConfig()
    print config.get_value('RECIPE_LOG_PATH', "20140525")

    # print config.get("DEFAULT", 'INDATA_PATH', 0, dict(UTDATE="20140525"))

    # print config.get("DEFAULT", 'QA_PATH', 0, dict(UTDATE="20140525"))
