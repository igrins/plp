import ConfigParser


class IGRINSConfig(object):
    def __init__(self, config_file=None):

        if config_file is None:
            config_file = 'recipe.config'
        self.config_file = config_file

        self.config = ConfigParser.ConfigParser()
        self.config.read(config_file)

    def get_value(self, option, utdate):
        return self.config.get("DEFAULT", option, 0, dict(UTDATE=utdate))


if __name__ == "__main__":

    igrins_config = IGRINSConfig()
    print igrins_config.get_value('INDATA_PATH', "20140525")

    # print config.get("DEFAULT", 'INDATA_PATH', 0, dict(UTDATE="20140525"))

    # print config.get("DEFAULT", 'QA_PATH', 0, dict(UTDATE="20140525"))
