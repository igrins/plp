import logging

logger = logging


def set_level(level):
    "for now, ignore level and set debug"
    # print("setting log level to debug")
    logging.basicConfig(level=level)


def info(msg):
    ""
    logger.info(msg)


def debug(msg):
    ""
    logger.debug(msg)
