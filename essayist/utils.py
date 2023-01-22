import logging
import re
import sys

URL_PATTERN = re.compile(
    r"https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)"
)
NEWLINE_PATTERN = re.compile(r"\s*\n\s*")
SPACE_PATTERN = re.compile(r" +")
ID_TAG_PATTERN = re.compile(r"[@#][a-zA-Z가-힣_.]+")
SNS_PATTERN = re.compile(r"Insta|insta|인스타|facebook|페이스북|페북|brunch|브런치")
SOURCE_PATTERN = re.compile(r"출처|©|by")
BULLET_PATTERN = re.compile(r"^[※*-▶]")
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}")
JOB_PATTERN = re.compile(r"일러스트레이터|칼럼|글쓴이|매거진|출판사|작가|(글|사진|그림)[\s.:]")
DATE_PATTERN = re.compile(r"(20)?\d{2}\.\s?\d{1,2}\.\s?\d{1,2}|(20)?\d{2}년[.\s]\d{1,2}월")


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


def filter_condition(text: str, long_length: int = 200, short_length: int = 40) -> bool:
    return (
        len(text) < long_length
        and (
            URL_PATTERN.search(text)
            or ID_TAG_PATTERN.search(text)
            or BULLET_PATTERN.fullmatch(text)
            or EMAIL_PATTERN.search(text)
        )
        or (
            len(text) < short_length
            and (
                SNS_PATTERN.search(text)
                or JOB_PATTERN.search(text)
                or DATE_PATTERN.search(text)
                or SOURCE_PATTERN.search(text)
            )
        )
    )


def normalize_text(text: str, filter_last_n: int = 10) -> str:
    text = NEWLINE_PATTERN.sub("\n", text)
    text = SPACE_PATTERN.sub(" ", text)
    sentences = text.split("\n")
    last_sentences = [sent for sent in sentences[-filter_last_n:] if not filter_condition(sent)]
    text = "\n".join(sentences[:-filter_last_n] + last_sentences)
    text = URL_PATTERN.sub("", text)
    text = SPACE_PATTERN.sub(" ", text)
    return text.strip()
