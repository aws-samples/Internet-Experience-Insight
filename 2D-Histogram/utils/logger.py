import sys
import logging
import colorlog

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.WARNING)

color_formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s: %(message)s",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
)

# StreamHandler to output logs to stdout with color
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(color_formatter)
_logger.addHandler(stream_handler)

class _LoggerConfig:
    def __init__(self):
        self.err_cnt = 0
        self.critical_cnt = 0

_config = _LoggerConfig()


def log_info(msg: str):
    _logger.info(msg)


def log_critical(msg: str):
    _config.critical_cnt += 1
    _logger.critical(msg)


def log_err(msg: str):
    _config.err_cnt += 1
    print("")
    if sys.exc_info()[0] is not None:
        _logger.exception(msg)
    else:
        _logger.error(msg)
    print("")


def get_err_cnt() -> int:
    return _config.err_cnt


def get_critical_cnt() -> int:
    return _config.critical_cnt


def init():
    _config.err_cnt = 0
    _config.critical_cnt = 0


def log_warning(msg: str):
    print("")
    _logger.warning(msg)
    print("")