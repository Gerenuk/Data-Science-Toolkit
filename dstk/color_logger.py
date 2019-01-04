import sys
import logging
import colorlog


def color_logger(
    file=sys.stdout,
    name=None,
    level="INFO",
    log_format="%(log_color)s[%(levelname)s]%(white)s %(asctime)s %(funcName)s(L%(lineno)s)%(reset)s: %(message)s", # %(name)s (e.g. root)
    date_format="%H:%M:%S", # %d.%m.%y
):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if isinstance(file, str):
        handler = logging.FileHandler(file)
    else:
        handler = logging.StreamHandler(file)

    formatter = colorlog.ColoredFormatter(log_format, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
