import logging

logger = logging.getLogger("IGRINS")

logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)

logger.addHandler(ch)


def set_level(level):
    "for now, ignore level and set debug"
    print("setting log level to debug: ", level)
    ch.setLevel(level)


def debug(msg):
    ""
    logger.debug(msg)


def info(msg):
    ""
    logger.info(msg)


def warning(msg):
    ""
    logger.warning(msg)
