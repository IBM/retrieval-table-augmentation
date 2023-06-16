import random

from util.line_corpus import write_open
from util.args_help import fill_from_args
import ujson as json
import os
import logging

logger = logging.getLogger(__name__)


class Options:
    def __init__(self):
        self.table_dir = ''
        self.split_definitions = ''
        self.passage_file = ''
        self.query_dir = ''


def cell_text(cell):
    if is_entity(cell):
        page, label = cell[1:-1].split('|')
        page = page.replace('_', ' ')
        """
        if page.lower() == label.lower():
            return label
        else:
            return f'{label} ({page})'
        """
        return page  # I think the point of Entitables is we predict the page
    return cell


def is_entity(cell):
    return cell.startswith('[') and cell.endswith(']') and cell.find('|') >= 0 and len(cell[1:-1].split('|')) == 2


def make_column_query(table_id, title, header, rows, num_seeds: int) -> dict:
    title = f'COLUMN {title}'
    num_cell_values = 3
    text = ' * '.join([h + ': ' + ', '.join([row[ndx] for row in rows[:num_cell_values]])
                       for ndx, h in enumerate(header[:num_seeds])])
    text = text + ' * COLUMN'
    answers = header[num_seeds:]  # FIXME: .lower()
    return {'id': f'{table_id}::COL{num_seeds}', 'table_id': table_id, 'title': title, 'text': text, 'answers': answers}


def make_row_query(table_id, title, header, rows, num_seeds: int) -> dict:
    title = f'ROW {title}'
    text0 = (header[0] + ': ') + ', '.join([row[0] for row in rows[:num_seeds]]) + ', ROW'
    text = ' * '.join([h + ': ' + ', '.join([row[ndx+1] for row in rows[:num_seeds]])
                       for ndx, h in enumerate(header[1:])])
    text = text0 + ' * ' + text
    answers = [row[0] for row in rows[num_seeds:]]  # FIXME: only row entities
    return {'id': f'{table_id}::ROW{num_seeds}', 'table_id': table_id, 'title': title, 'text': text, 'answers': answers}


def qid2table_id(qid):
    return qid.split('::')[0]


def table2query(jobj, num_seeds: int, row_or_column: str) -> dict:
    """

    :param jobj:
    :param num_seeds:
    :param row_or_column:
    :return: dict with id, table_id, title, text, answers
    """
    row_or_column = row_or_column.lower()[:1]
    # make column query or make row query
    table_id = jobj['table_id']
    title = jobj['title']
    header = jobj['header']
    rows = jobj['rows']
    row_entities = jobj['row_entities']
    header_entities = jobj['header_entities']
    """
    FIXME: looks like column_id_validation often does not have header entities!
    if row_or_column != 'r' and not header_entities[0] and all(header_entities[1:num_seeds + 2]):
        col_q = True
        num_seeds += 1
    elif all(header_entities[:num_seeds + 1]) and row_or_column != 'r':
        col_q = True
    else:
        col_q = False
    """
    col_q = row_or_column != 'r'
    row_q = all(row_entities[:num_seeds + 1]) and row_or_column != 'c'
    if (col_q and row_q and random.randint(0, 1) == 0) or col_q:
        query = make_column_query(table_id, title, header, rows, num_seeds)
    elif row_q:
        query = make_row_query(table_id, title, header, rows, num_seeds)
    else:
        return None
    return query


def entity_passages(table_id, title, header, header_ents, rows, raw_rows):
    row_ents_all = [[is_entity(cell) for cell in row] for row in raw_rows]
    passages = []
    # header passage
    if any(header_ents):
        pid = f'{table_id}::header'
        first_rows = [', '.join([row[cndx] for row in rows[:3]]) for cndx in range(len(header))]
        text = ' * '.join([f'{head}: {cells}' for head, cells in zip(header, first_rows)])
        passages.append({'pid': pid, 'title': title, 'text': text})
    # passage for rows that contain an entity
    # CONSIDER: maybe do 3 row blocks
    for row_ndx, row in enumerate(rows):
        if any(row_ents_all[row_ndx]):
            pid = f'{table_id}::{row_ndx}'
            text = ' * '.join([f'{header}: {cell}' for header, cell in zip(header, row)])
            passages.append({'pid': pid, 'title': title, 'text': text})
    return passages


def row_passages(table_id, title, header, rows):
    passages = []
    for row_ndx, row in enumerate(rows):
        pid = f'{table_id}::{row_ndx}'
        text = ' * '.join([f'{header}: {cell}' for header, cell in zip(header, row)])
        passages.append({'pid': pid, 'title': title, 'text': text})
    return passages


def main():
    opts = Options()
    fill_from_args(opts)
    test_tables = dict()  # id2split
    for file in os.listdir(opts.split_definitions):
        with open(os.path.join(opts.split_definitions, file)) as f:
            test_table_list = json.load(f)
            for table_id in test_table_list:
                test_tables[table_id] = file[:-5]

    test_splits = list(set(test_tables.values()))
    test_splits.sort()
    assert test_splits == ['column_id_test',  'column_id_validation', 'row_id_test',  'row_id_validation']
    split2out = {split_name: write_open(os.path.join(opts.query_dir, f'{split_name}.jsonl'))
                 for split_name in test_splits + ['train']}
    with write_open(opts.passage_file) as out:
        for file in os.listdir(opts.table_dir):
            with open(os.path.join(opts.table_dir, file)) as f:
                tables = json.load(f)
            for table_id, table in tables.items():
                header_ents = [is_entity(h) for h in table['title']]
                header = [cell_text(h) for h in table['title']]
                title = table['pgTitle'] + '\n' + table['caption']
                rows = [[cell_text(cell) for cell in row] for row in table['data']]
                row_ents = [is_entity(row[0]) for row in table['data']] \
                    if len(table['data']) > 0 and len(table['data'][0]) > 0 else []
                assert all([len(row) == len(header) for row in rows])
                if table_id in test_tables:
                    split = test_tables[table_id]
                    qout = split2out[split]
                    qout.write(json.dumps({'table_id': table_id, 'title': title, 'header': header, 'rows': rows,
                                          'row_entities': row_ents, 'header_entities': header_ents})+'\n')
                    continue
                # CONSIDER: filters should apply for train query tables, but maybe not passage tables
                if len(header) < 4:
                    continue
                if len(rows) < 4:
                    continue
                if any([h == header[0] for h in header[1:]]):
                    continue
                # write out train queries
                if all(row_ents[:3]) or all(header_ents[:3]):
                    qout = split2out['train']
                    qout.write(json.dumps({'table_id': table_id, 'title': title, 'header': header, 'rows': rows,
                                           'row_entities': row_ents, 'header_entities': header_ents}) + '\n')
                for psg in row_passages(table_id, title, header, rows):
                    out.write(json.dumps(psg)+'\n')
    for out in split2out.values():
        out.close()


if __name__ == "__main__":
    main()