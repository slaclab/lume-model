import logging
import logging.config
import logging.handlers
import os


def is_debug():
    """
    Check for environment variable to be debug or not

    Returns:
    bool

    """
    if os.environ.get("LOGLEVEL") == "DEBUG":
        return True


def configure():
    """
    Configures file and console loggers
    """

    log_handlers = {
        "console": {"class": "logging.StreamHandler", "formatter": "standard"},
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "filename": "lume-model.log",
        },
    }

    # configure python loggers
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": True,
            "loggers": {
                "": {
                    "level": "DEBUG" if is_debug() else "WARN",
                    "handlers": ["console", "file"],
                },
                "lume-model": {
                    "level": "DEBUG" if is_debug() else "INFO",
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
            },
            "handlers": log_handlers,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                },
            },
        }
    )
