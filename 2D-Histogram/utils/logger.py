import sys
import logging
import colorlog

_logger = logging.getLogger()
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

_err_cnt = 0
_critical_cnt = 0


def log_info(msg: str):
    print(msg)
    _logger.info(msg)


def log_critical(msg: str):
    global _critical_cnt
    _critical_cnt += 1
    _logger.critical(msg)


def log_err(msg: str):
    global _err_cnt
    _err_cnt += 1
    print("")
    if sys.exc_info()[0] is not None:
        _logger.exception(msg)
    else:
        _logger.error(msg)
    print("")


def get_err_cnt() -> int:
    return _err_cnt


def get_critical_cnt() -> int:
    return _critical_cnt


def init():
    global _err_cnt, _critical_cnt
    _err_cnt = 0
    _critical_cnt = 0


def log_warning(msg: str):
    print("")
    _logger.warning(msg)