import contextlib
import json
import os
import string

import numpy as np
from dateutil.parser import DEFAULTPARSER

from util.args_help import fill_from_args

HOME = {
    "SL": "/home/xueqing",
    "CCC": "/u/xueqingwu0",
}[os.environ['SERVER']]


def fill_from_args_(defaults):
    fill_from_args(defaults)
    to_print = {}
    for k, v in vars(defaults).items():
        try:
            json.dumps(v)
            to_print[k] = v
        except:
            to_print[k] = str(v)
    print(json.dumps(to_print, indent=2))
    return defaults


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def truncate(text: str, char_limit: int):
    # CONSIDER textwrap.shorten
    if len(text) <= char_limit:
        return text
    text = text.strip()
    words = text.split(' ')
    cur_len = -1  # minus one since we don't begin with a space
    trunc_words = []
    for word in words:
        cur_len += 1 + len(word)
        if cur_len > char_limit:
            break
        trunc_words.append(word)
    if len(trunc_words) == 0:
        return text[:char_limit]
    else:
        return ' '.join(trunc_words)


def normalize_value(x):
    def normalize_date(x):
        d, _ = DEFAULTPARSER._parse(x)
        if d is None:
            return x

        if int(d.year is not None and d.year > 0) + int(d.month is not None and d.month > 0) + \
                int(d.day is not None and d.day > 0) <= 1:
            return x

        ret = ""
        if d.year is not None:
            if len(str(d.year)) != 4:
                return x
            ret = "-" + str(d.year)
        if d.month is not None:
            ret += "-{:02d}".format(d.month)
        if d.day is not None:
            ret += "-{:02d}".format(d.day)
        return ret[1:]

    if "(age" in x:
        x = x.split("(age")[0].strip()

    return normalize_date(x)


def to_passages_a(train_table, k_rows=3, normalizer=None):
    # FIXME: remove this and the *_dump_passages.py
    """
    this version breaks tables into chunks of rows that contain at least one entity
    :param train_table:
    :param k_rows:
    :return:
    """
    row_pops = []
    header = train_table['header']
    rows = train_table['rows']
    row_reps = []
    rstart = 0
    for rndx in range(len(rows)):
        row = rows[rndx]
        if normalizer is not None:
            row = [normalizer(cell) for cell in row]
        cell_strs = [f'«{cell}»' if len(cell) > 0 else "" for cell in row]
        # wrap all cells but empty ones: numbers are not cells but should be filled
        row_rep = ' * '.join([f'{header[cndx]}: {cell_strs[cndx]}' for cndx in range(len(row))])
        row_reps.append(row_rep)
        if len(row_reps) >= k_rows:
            pid = train_table['table_id'] + f':R[{rstart}-{rndx}]'
            row_pops.append({'pid': pid, 'title': train_table['title'], 'text': ' | '.join(row_reps)})
            row_reps = []
            rstart = rndx + 1
    if len(row_reps) > 0:
        pid = train_table['table_id'] + f':R[{rstart}-{len(rows)}]'
        row_pops.append({'pid': pid, 'title': train_table['title'], 'text': ' | '.join(row_reps)})
    return row_pops


def get_date_type(x):
    d, _ = DEFAULTPARSER._parse(x)
    if d is None:
        return None

    if int(d.year is not None) + int(d.month is not None) + int(d.day is not None) > 1:  # date
        if d.year is not None and not 1000 <= d.year <= 3000:  # very wide range
            return None
        if d.month is not None and not d.month in list(range(1, 13)):
            return None
        if d.day is not None and not 1 <= d.day <= 31:
            return None
        return "date"

    if int(d.hour is not None) + int(d.minute is not None) + int(d.second is not None) + \
            int(d.microsecond is not None) > 1:  # time
        return "time"

    return None


def get_type(x):
    date_type = get_date_type(x)
    if date_type is not None:
        return "date-time"  # date or time
    elif not any(xx in string.ascii_letters for xx in x):
        return "digits"
    else:
        return "string"
