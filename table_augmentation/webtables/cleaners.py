import unicodedata
from table_augmentation.table import Table
import numpy as np


def clean(tbl: Table) -> None:
    unicode_clean(tbl)
    normalize_empty(tbl)
    empty_clean(tbl)
    tbl.validate()


def unicode2ascii(text: str) -> str:
    """
    normalize unicode to ascii
    :param text:
    :return:
    """
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii').strip()


def unicode_clean(tbl: Table) -> None:
    tbl.title = unicode2ascii(tbl.title)
    tbl.header = [unicode2ascii(h) for h in tbl.header]
    tbl.rows = [[unicode2ascii(cell) for cell in row] for row in tbl.rows]


def normalize_empty(tbl: Table) -> None:
    empty_markers = ['-', '--', '?']
    tbl.header = ['' if h in empty_markers else h for h in tbl.header]
    tbl.rows = [['' if cell in empty_markers else cell for cell in row] for row in tbl.rows]


def empty_clean(tbl: Table) -> None:
    # remove empty columns, rows
    remove_cols = []
    for col_ndx in range(len(tbl.header)):
        if len(tbl.header[col_ndx]) == 0 or all(len(row[col_ndx]) == 0 for row in tbl.rows):
            remove_cols.append(col_ndx)

    # find new key column ndx
    cand_cols = [cndx for cndx in range(len(tbl.header)) if tbl.candidate_cells[0, cndx]]
    assert len(cand_cols) <= 1
    if len(cand_cols) == 1:
        key_col_ndx = cand_cols[0]
        new_key_col_ndx = key_col_ndx
        for rc in remove_cols:
            if rc == key_col_ndx:
                new_key_col_ndx = -1
            elif new_key_col_ndx >= 0 and rc < key_col_ndx:
                new_key_col_ndx -= 1
    else:
        new_key_col_ndx = -1

    rows = [[cell for cndx, cell in enumerate(row) if cndx not in remove_cols] for row in tbl.rows]
    tbl.header = [h for cndx, h in enumerate(tbl.header) if cndx not in remove_cols]
    tbl.rows = [row for row in rows if not all(len(cell) == 0 for cell in row)]

    tbl.candidate_cells = np.zeros((len(tbl.rows), len(tbl.header)), dtype=np.int8)
    if 0 <= new_key_col_ndx < len(tbl.header):
        tbl.candidate_cells[:, new_key_col_ndx] = 1


