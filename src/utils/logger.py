import logging


_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")


def get_logger():
    return _logger

def suppress_matplotlib_debug():
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
