from enum import Enum
from typing import Callable, List, Optional, Dict
from table_augmentation.table import Table
import functools
from table_augmentation.normalization import NormalizationStyle
import tabulate


class Query:
    __slots__ = ['qid', 'table_id', 'title', 'text', 'answers']

    def __init__(self, qid: str, table_id: str, title: str, text: str, answers: List[str]):
        self.qid = qid
        self.table_id = table_id
        self.title = title
        self.text = text
        self.answers = answers
        # NOTE: we allow blank answers for cell filling
        # if not all(len(a) > 0 for a in self.answers):
        #     print(f'WARNING: bad answers for {qid} on {table_id}: {self.answers}')

    def to_dict(self):
        return {'qid': self.qid, 'table_id': self.table_id,
                'title': self.title, 'text': self.text,
                'answers': self.answers}

    @staticmethod
    def from_dict(jobj: dict):
        id_ = jobj['qid'] if 'qid' in jobj else jobj['id']
        table_id = jobj['table_id'] if 'table_id' in jobj else jobj['table']['table_id']
        # answers = [a for a in jobj['answers'] if len(a) > 0]
        # if len(answers) == 0:
        #     return None
        return Query(id_, table_id, jobj['title'], jobj['text'], jobj['answers'])


def make_row_query(table: Table, num_seeds: int, answer_normalize: Callable[[str], str], no_answers=False) -> List[Query]:
    title_prefix = 'ROW '  # TODO: probably should just be empty string
    answer_prompt = 'ROW'  # TODO: maybe [MASK]?

    if (not no_answers and len(table.rows) <= num_seeds) or len(table.header) == 0 or table.key_col_ndx == -1:
        return []

    if no_answers:
        answers = []
    else:
        answers = set([answer_normalize(row[table.key_col_ndx]) for row, re in
                       zip(table.rows[num_seeds:], table.candidate_cells[num_seeds:, table.key_col_ndx]) if re])
        answers = [a for a in answers if len(a) > 0]
        if len(answers) == 0:
            return []

    title = title_prefix + table.title
    text = ' * '.join([h + ': ' + ', '.join([str(row[ndx]) for row in table.rows[:num_seeds]] +
                                            ([answer_prompt] if ndx == table.key_col_ndx else []))
                       for ndx, h in enumerate(table.header)])

    return [Query(f'{table.table_id}::ROW{num_seeds}', table.table_id, title, text, answers)]


def make_col_query(table: Table, num_seeds: int, answer_normalize: Callable[[str], str], no_answers=False) -> List[Query]:
    title_prefix = 'COLUMN '  # TODO: probably should just be empty string
    answer_prompt = 'COLUMN'  # TODO: maybe [MASK]?
    num_cell_values = 3

    if not no_answers and len(table.header) <= num_seeds:
        return []

    title = title_prefix + table.title
    text = ' * '.join([h + ': ' + ', '.join([str(row[ndx]) for row in table.rows[:num_cell_values]])
                       for ndx, h in enumerate(table.header[:num_seeds])])

    text = text + ' * ' + answer_prompt
    if no_answers:
        answers = []
    else:
        answers = list(set([answer_normalize(h) for h in table.header[num_seeds:]]))
        invalid_answer = sum([1 for a in answers if type(a) is not str or "unnamed" in a.lower()])
        if invalid_answer == 0:
            answers = [a for a in answers if len(a) > 0]
        else:
            # print("Cannot make query col augmentation for table: ", table.header)
            return []
        if len(answers) == 0:
            return []
    return [Query(f'{table.table_id}::COL{num_seeds}', table.table_id, title, text, answers)]


def make_cell_query(table: Table, num_query_rows: int, answer_normalize: Callable[[str], str]) -> List[Query]:
    # NOTE: unused currently
    def make_query(table, query_row_id, query_col_id, context_rows):
        row_ids = [query_row_id, ] + list(context_rows)
        assert len(set(row_ids)) == len(row_ids)

        table = {
            k: [v[i] for i in row_ids] if k in ["rows", "row_entities", ] and isinstance(v, list) else v
            for k, v in table.items()
        }
        answers = [table["rows"][0][query_col_id], ]
        table["rows"][0][query_col_id] = "[MASK]"

        # text = ' * '.join([h + ': ' + ', '.join([row[ndx] for row in rows]) for ndx, h in enumerate(header)])
        # NOTE: use row population's format. Makes more sense
        text = " | ".join([" * ".join([h + ': ' + c for h, c in zip(table["header"], row)]) for row in table["rows"]])
        return {
            'title': table["title"], 'text': text, 'answers': answers
        }

    def make_query2(table: Table, query_row_id: int, query_col_id: int, context_rows: List[int]):
        row_ids = [query_row_id, ] + list(context_rows)
        assert len(set(row_ids)) == len(row_ids)
        rows = [table.rows[i] for i in row_ids]
        answer = rows[0][query_col_id]
        rows[0][query_col_id] = "[MASK]"
        text = " | ".join([" * ".join([h + ': ' + c for h, c in zip(table.header, row)]) for row in rows])
        return table.title, text, answer

    # entitable test-style query making
    # FIXME: if tables.answers is None use the logic in table_augmentation.cell_filling.bm25_dpr_training_data.make_training_queries
    answers = sorted(table.answers)
    rows_to_fill = [x[0] for x in answers]
    ret = []
    for row_id, col_id, gold_answers in answers:  # gold_answers not used
        context_rows = []
        for i in range(row_id + 1, len(table.rows)):
            if i not in rows_to_fill:
                context_rows.append(i)
                if len(context_rows) == num_query_rows - 1:
                    break
        qid = "{}-[{}]-{}".format(
            table.table_id, "-".join([str(i) for i in [row_id, ] + context_rows]), col_id
        )
        # query = make_query({k: getattr(table, k) for k in table.__slots__}, row_id, col_id, context_rows)
        # ret.append(Query(qid, table.table_id, query['title'], query['text'], [answer_normalize(query['answers'][0]), ]))
        title, text, answer = make_query2(table, row_id, col_id, context_rows)
        ret.append(Query(qid, table.table_id, title, text, [answer_normalize(answer)]))
    return ret


def answer_candidates(text: str) -> List[str]:
    end_ndx = -1
    answers = []
    while True:
        start_ndx = text.find('«', end_ndx + 1)
        if start_ndx == -1:
            return answers
        end_ndx = text.find('»', start_ndx + 1)
        assert end_ndx != -1
        answers.append(text[start_ndx + 1:end_ndx])


def is_answer_bearing(text: str, normalized_answers: List[str], answer_normalize: Callable[[str], str]) -> bool:
    for ans in answer_candidates(text):
        if answer_normalize(ans) in normalized_answers:
            return True
    return False


class Passage:
    __slots__ = ['pid', 'title', 'text']

    def __init__(self, pid: str, title: str, text: str):
        self.pid = pid
        self.title = title
        self.text = text

    def is_answer_bearing(self, normalized_answers: List[str], answer_normalize: Callable[[str], str]) -> bool:
        return is_answer_bearing(self.text, normalized_answers, answer_normalize)

    def answer_candidates(self) -> List[str]:
        return answer_candidates(self.text)

    @staticmethod
    def from_dict(jobj: Dict[str, str]):
        return Passage(jobj['pid'], jobj['title'], jobj['text'])

    def to_dict(self) -> Dict[str, str]:
        return {'pid': self.pid, 'title': self.title, 'text': self.text}

    def get_table_id(self):
        tbl_end = self.pid.rfind(':')  # NOTE: we require that pid suffixes not include ':'
        if tbl_end == -1:  # old style column population passages have pid == table_id
            return self.pid
        assert 0 <= tbl_end < len(self.pid)
        return self.pid[:tbl_end]


def make_row_passages(table: Table, k_rows: int = 3) -> List[Passage]:
    """
    this version breaks tables into chunks of rows that contain at least one entity
    :param table:
    :param k_rows:
    :return:
    """
    row_pops = []
    row_reps = []
    rstart = 0
    for rndx, row in enumerate(table.rows):
        if table.candidate_cells[rndx, :].sum() == 0:
            continue
        cell_strs = [f'«{cell}»' if table.candidate_cells[rndx, cndx] else cell for cndx, cell in enumerate(row)]
        row_rep = ' * '.join([f'{table.header[cndx]}: {cell_strs[cndx]}' for cndx in range(len(row))])
        row_reps.append(row_rep)
        if len(row_reps) >= k_rows:
            pid = table.table_id + f':R[{rstart}-{rndx}]'
            row_pops.append(Passage(pid, table.title, ' | '.join(row_reps)))
            row_reps = []
            rstart = rndx + 1
    if len(row_reps) > 0:
        pid = table.table_id + f':R[{rstart}-{len(table.rows)}]'
        row_pops.append(Passage(pid, table.title, ' | '.join(row_reps)))
    return row_pops


def tabulate_row_passage(passage: Passage) -> str:
    headed_rows = [row.split(' * ') for row in passage.text.split(' | ')]
    header = [cell[:cell.find(':')] for cell in headed_rows[0]]
    rows = [[cell[cell.find(':')+1:] for cell in row] for row in headed_rows]
    return passage.title + '\n' + tabulate.tabulate(rows, headers=header)


def make_col_passages(table: Table, k_rows: int = 2) -> List[Passage]:
    examples = [', '.join([table.rows[rndx][cndx] for rndx in range(k_rows)]) for cndx in range(len(table.header))]
    header_rep = ' * '.join([f'«{h}»: {ex}' for h, ex in zip(table.header, examples)])
    return [Passage(table.table_id+':A', table.title, header_rep)]


def tabulate_col_passage(passage: Passage) -> str:
    cols = [(col[:col.find(':')], col[col.find(':')+1:].split(', ')) for col in passage.text.split(' * ')]
    header = [h for h, vs in cols]
    col_vals = [vs for h, vs in cols]
    rows = list(zip(*col_vals))
    # '\n' + passage.text + '\n' + str(header) + '\n' + str(rows) +
    return passage.title + '\n' + tabulate.tabulate(rows, headers=header)


def make_cell_passages(table: Table, k_rows: int = 3) -> List[Passage]:
    """
    this version breaks tables into chunks of rows where all non-empty cells are candidates
    :param table:
    :param k_rows:
    :return:
    """
    row_pops = []
    row_reps = []
    rstart = 0
    for rndx in range(len(table.rows)):
        row = table.rows[rndx]
        cell_strs = [f'«{cell}»' if len(cell) > 0 else "" for cell in row]
        # wrap all cells but empty ones: numbers are not cells but should be filled
        row_rep = ' * '.join([f'{table.header[cndx]}: {cell_strs[cndx]}' for cndx in range(len(row))])
        row_reps.append(row_rep)
        if len(row_reps) >= k_rows:
            pid = table.table_id + f':R[{rstart}-{rndx}]'
            row_pops.append(Passage(pid, table.title, ' | '.join(row_reps)))
            row_reps = []
            rstart = rndx + 1
    if len(row_reps) > 0:
        pid = table.table_id + f':R[{rstart}-{len(table.rows)}]'
        row_pops.append(Passage(pid, table.title, ' | '.join(row_reps)))
    return row_pops


def tabulate_cell_passage(passage: Passage) -> str:
    # TODO
    raise NotImplementedError


class AugmentationTask(Enum):
    row = 1
    col = 2
    cell = 3


class TaskOptions:
    def __init__(self, task=None, no_answers=False, num_seeds=2, answer_normalization=None, num_passage_rows=3):
        self.task = AugmentationTask.col if task is None else task
        self.no_answers = no_answers
        self.num_seeds = num_seeds
        self.answer_normalization = \
            NormalizationStyle.identity if answer_normalization is None else answer_normalization
        # TODO: a _post_argparse that warns if we have a normalization at odds with task

        self.num_passage_rows = num_passage_rows
        # CONSIDER: cell_normalization? header_normalization?

        self._answer_normalizer = None
        self._required_args = ['task']

    def get_query_maker(self) -> Callable[[Table], List[Query]]:
        """
        returns a function that creates a Query from a Table if possible, returns None otherwise
        :param answer_normalize: a function from strings to string to normalize the answer(s)
        :param num_seeds: for row and column population the number of seed rows or columns
        :return: Query if a query for this task can be created, otherwise None
        """
        if self._answer_normalizer is None:
            self._answer_normalizer = self.answer_normalization.get_normalizer()
        if self.task == AugmentationTask.row:
            return functools.partial(make_row_query, answer_normalize=self._answer_normalizer,
                                     num_seeds=self.num_seeds, no_answers=self.no_answers)
        elif self.task == AugmentationTask.col:
            return functools.partial(make_col_query, answer_normalize=self._answer_normalizer,
                                     num_seeds=self.num_seeds, no_answers=self.no_answers)
        elif self.task == AugmentationTask.cell:
            return functools.partial(make_cell_query, answer_normalize=self._answer_normalizer,
                                     num_query_rows=self.num_seeds)
        else:
            raise NotImplementedError

    def get_passage_maker(self) -> Callable[[Table], List[Passage]]:
        if self.task == AugmentationTask.row:
            return functools.partial(make_row_passages, k_rows=self.num_passage_rows)
        elif self.task == AugmentationTask.col:
            return functools.partial(make_col_passages, k_rows=self.num_passage_rows)
        elif self.task == AugmentationTask.cell:
            return functools.partial(make_cell_passages, k_rows=self.num_passage_rows)
        else:
            raise NotImplementedError

    def get_passage_displayer(self) -> Callable[[Passage], str]:
        if self.task == AugmentationTask.row:
            return tabulate_row_passage
        elif self.task == AugmentationTask.col:
            return tabulate_col_passage
        elif self.task == AugmentationTask.cell:
            return tabulate_cell_passage
        else:
            raise NotImplementedError