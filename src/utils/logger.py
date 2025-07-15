import logging

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

file_handler = logging.FileHandler("app.log", mode="w")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
_logger.addHandler(file_handler)


def get_logger():
    return _logger


def suppress_matplotlib_debug():
    _logger.debug("Suppressing matplotlib font manager logger")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
