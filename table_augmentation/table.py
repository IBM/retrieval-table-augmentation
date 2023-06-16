from typing import List, Set, Callable, Optional, Tuple
import numpy as np
import tabulate


def _entitable_is_entity(cell):
    return cell.startswith('[') and cell.endswith(']') and cell.find('|') >= 0 and len(cell[1:-1].split('|')) == 2


def _entitable_cell_text(cell: str, max_char_cell: int):
    if _entitable_is_entity(cell):
        page, label = cell[1:-1].split('|')
        page = page.replace('_', ' ')
        assert '«' not in page
        assert '»' not in page
        # if len(page) > 100:
        #    print(f'Long entity name: {page} ({len(page)})')
        return page  # the point of EntiTables is we predict the page
    return truncate(cell, max_char_cell)  #.replace('«', '<').replace('»', '>') we will do the replacement when making passages


def truncate(text, char_limit: int):
    if type(text) is not str:
        return text
    # TODO: textwrap.shorten
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


class Table:
    __slots__ = ['table_id', 'title', 'header', 'rows', 'candidate_cells', 'key_col_ndx', "answers"]

    @staticmethod
    def from_dict(jobj: dict):
        header = jobj['header']
        rows = jobj['rows'] if 'rows' in jobj else []
        if any(len(header) != len(row) for row in rows):
            raise ValueError(f'bad table: {jobj}')
        key_col_ndx = jobj['key_col_ndx'] if 'key_col_ndx' in jobj else 0
        if 'candidate_cells' in jobj:
            candidate_cells = np.array(jobj['candidate_cells'], dtype=np.int8)
        else:
            candidate_cells = np.zeros((len(rows), len(header)), dtype=np.int8)
            if len(rows) > 0 and 0 <= key_col_ndx < len(header):
                candidate_cells[:, key_col_ndx] = 1
        table_id = jobj['table_id'] if 'table_id' in jobj else '::NA'
        title = jobj['title'] if 'title' in jobj else ''
        answers = jobj.get("answers", None)
        return Table(table_id, title, header, rows, key_col_ndx, candidate_cells, answers)

    @staticmethod
    def from_webtable(jobj: dict):
        table_id = jobj['filename']
        title = jobj['pageTitle']
        if jobj['title']:
            if title:
                title = title + ' | ' + jobj['title']
            else:
                title = jobj['title']
        cols = jobj['relation']
        head_row_ndx = jobj['headerRowIndex']
        if head_row_ndx != -1:
            header = [col[head_row_ndx] for col in cols]
        else:
            return None
        rows = [[col[rndx] for col in cols]
                for rndx in range(len(cols[0])) if rndx != head_row_ndx]
        key_col_ndx = jobj['keyColumnIndex']
        candidate_cells = np.zeros((len(rows), len(header)), dtype=np.int8)
        if 0 <= key_col_ndx < len(header):
            candidate_cells[:, key_col_ndx] = 1
        return Table(table_id, title, header, rows, key_col_ndx, candidate_cells)

    """
    {
      "file_name": "https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/A0329/CSV/1.0/en",
      "dataset_id": "d6c7ba9d-d761-4c86-bac6-7dff7841e534",
      "dataset_name": "A0329 - 1996 Persons in Private Households",
      "dataset_description": "1996 Persons in Private Households",
      "table_id": "8b25036c-1080-42c6-b86f-b102ba775382",
      "table_name": "1996 Persons in Private Households",
      "table_description": "",
      "tags": [
        "a0329",
        "census-1996",
        "cso"
      ],
      "organization_id": "central-statistics-office",
      "organization_name": "central-statistics-office. Central Statistics Office",
      "column_headers": [
        {
          "name": "STATISTIC CODE",
          "dtype": "object"
        },
        {
          "name": "Statistic",
          "dtype": "object"
        },
      ],
      "rows": 3420,
      "sample_data": [
        [
          "A0329",
          "1996 Persons in Private Households",
        ],
      ],
      "_id": "3:0"
    }
    """

    @staticmethod
    def from_ckan(jobj: dict, max_char_cell=50, max_char_title=100):
        table_id = jobj['dataset_id'] + ":" + jobj['table_id']
        title = jobj.get('dataset_name', "")
        key_col_ndx = jobj['key_col_ndx'] if 'key_col_ndx' in jobj else 0
        if jobj['table_name']:
            if title:
                title = jobj['table_name'] + ' | ' + title
            else:
                title = jobj['table_name']
        title = truncate(title, max_char_title)
        header = [truncate(c["name"], max_char_cell) for c in jobj['column_headers']]
        rows = [[truncate(v, max_char_cell) for v in r] for r in jobj["sample_data"]]
        candidate_cells = np.zeros((len(rows), len(header)), dtype=np.int8)
        if 0 <= key_col_ndx <= len(header):
            candidate_cells[:, key_col_ndx] = 1
        return Table(table_id, title, header, rows, key_col_ndx, candidate_cells)

    @staticmethod
    def from_entitable(table_id: str, table: dict, *, max_char_cell=50, max_char_title=100):
        header = [_entitable_cell_text(h, max_char_cell) for h in table['title']]
        full_title = table['pgTitle'] + ' ' + table['caption']
        title = truncate(full_title, max_char_title)
        rows = [[_entitable_cell_text(cell, max_char_cell) for cell in row] for row in table['data']]
        all_row_ents = [[1 if _entitable_is_entity(c) else 0 for c in row] for row in table['data']] \
                        if len(table['data']) > 0 and len(table['data'][0]) > 0 else [[]]
        # CONSIDER: should we also track entities present in the headers?
        return Table(table_id, title, header, rows, 0, np.array(all_row_ents, dtype=np.int8))

    def __init__(self, table_id: str, title: str, header: List[str], rows: List[List[str]],
                 key_col_ndx: int, candidate_cells: np.ndarray, answers: Optional[List[Tuple[int,int,str]]] = None):
        self.table_id = table_id
        self.title = title
        self.header = header
        self.rows = rows
        self.key_col_ndx = key_col_ndx
        self.candidate_cells = candidate_cells
        self.answers = answers  # NOTE: used only for cell filling, list of tuples of row_ndx, col_ndx, answer_string

    def validate(self):
        assert all(len(row) == len(self.header) for row in self.rows)

    def tabulate(self, *, num_rows=2) -> str:
        if num_rows < 0:
            num_rows = len(self.rows)
        # indicate key column with « »
        display_header = [h if hndx != self.key_col_ndx else '«' + h + '»' for hndx, h in enumerate(self.header)]
        return self.title + '\n' + tabulate.tabulate(self.rows[:num_rows], headers=display_header)

    def to_dict(self):
        return {'table_id': self.table_id, 'title': self.title, 'header': self.header, 'rows': self.rows,
                'key_col_ndx': self.key_col_ndx, 'candidate_cells': self.candidate_cells.tolist()}

    def _compute_key_col_ndx(self) -> int:
        """
        compute the key column as the one with the most candidates
        :return:
        """
        col_entities = [(cndx, self.candidate_cells[:, cndx].sum()) for cndx in range(len(self.header))]
        col_entities.sort(key=lambda x: x[1], reverse=True)
        key_col_ndx = col_entities[0][0]
        return key_col_ndx

    def answer_bearing_cells(self, normalized_answers: Set[str], normalize: Callable[[str], str], *,
                             all_candidates: bool = False) -> Set[str]:
        answers_found = set()
        for rndx, row in enumerate(self.rows):
            for cndx, cell in enumerate(row):
                if all_candidates or self.candidate_cells[rndx, cndx]:
                    nc = normalize(cell)
                    if nc in normalized_answers:
                        answers_found.add(nc)
        return answers_found

    def answer_bearing_header(self, normalized_answers: Set[str], normalize: Callable[[str], str]) -> Set[str]:
        answers_found = set()
        for h in self.header:
            nh = normalize(h)
            if nh in normalized_answers:
                answers_found.add(nh)
        return answers_found
