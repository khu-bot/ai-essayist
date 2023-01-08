import logging
import re
import sys

URL_PATTERN = re.compile(
    r"https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)"
)
NEWLINE_PATTERN = re.compile(r"\s*\n\s*")
SPACE_PATTERN = re.compile(r" +")


def get_logger(name: str) -> logging.Logger:
    """Return logger for logging

    Args:
        name: logger name
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
        logger.addHandler(handler)
    return logger


def normalize_text(text: str) -> str:
    text = URL_PATTERN.sub("", text)
    text = NEWLINE_PATTERN.sub("\n", text)
    text = SPACE_PATTERN.sub(" ", text)
    return text
