import ConfigParser
import warnings

default_config_content = """[DEFAULT]
REFDATE=20140316
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
        self.config_file = config_file

        self.config = ConfigParser.ConfigParser()

        import StringIO

        fp = StringIO.StringIO(default_config_content)
        self.config.readfp(fp)

        read_file = self.config.read(config_file)
        if not read_file:
            warnings.warn("no recipe.config is found. Internal default will be used.")

    def get_value(self, option, utdate):
        return self.config.get("DEFAULT", option, 0, dict(UTDATE=utdate))


if __name__ == "__main__":

    igrins_config = IGRINSConfig()
    print igrins_config.get_value('RECIPE_LOG_PATH', "20140525")

    # print config.get("DEFAULT", 'INDATA_PATH', 0, dict(UTDATE="20140525"))

    # print config.get("DEFAULT", 'QA_PATH', 0, dict(UTDATE="20140525"))
