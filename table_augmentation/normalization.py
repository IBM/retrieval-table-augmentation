from enum import Enum
import unicodedata
from typing import Callable


class NormalizationStyle(Enum):
    identity = 1
    lowercase = 2
    deunicode = 3
    # TODO: date and unit of measure normalize

    def get_normalizer(self) -> Callable[[str], str]:
        if self == NormalizationStyle.identity:
            return identity_normalize
        elif self == NormalizationStyle.lowercase:
            return lowercase_normalize
        elif self == NormalizationStyle.deunicode:
            return deunicode_normalize


def identity_normalize(text: str):
    return text


def lowercase_normalize(text: str):
    return text.lower()


def deunicode_normalize(text: str):
    """
    lowercase and normalize unicode to ascii, strip leading and trailing spaces
    :param text:
    :return:
    """
    return unicodedata.normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode('ascii').strip()
