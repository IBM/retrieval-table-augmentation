from util.line_corpus import write_open
from util.args_help import fill_from_args
import ujson as json
import os
import logging
import unicodedata

logger = logging.getLogger(__name__)


class Options:
    def __init__(self):
        # inputs
        self.table_dir = ''
        self.split_definitions = ''
        # outputs
        self.query_dir = ''
        self.passage_dir = ''
        # options
        self.max_char_title = 100
        self.max_char_cell = 50


def normalize(text: str):
    """
    lowercase and normalize unicode to ascii
    :param text:
    :return:
    """
    return unicodedata.normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode('ascii').strip()


def is_entity(cell):
    return cell.startswith('[') and cell.endswith(']') and cell.find('|') >= 0 and len(cell[1:-1].split('|')) == 2


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


def cell_text(cell: str, max_char_cell: int):
    if is_entity(cell):
        page, label = cell[1:-1].split('|')
        page = page.replace('_', ' ')
        assert '«' not in page
        assert '»' not in page
        # if len(page) > 100:
        #    print(f'Long entity name: {page} ({len(page)})')
        return page  # the point of EntiTables is we predict the page
    return truncate(cell, max_char_cell).replace('«', '<').replace('»', '>')


def table_id_from_pid(pid: str):
    tbl_end = pid.find(':')
    assert 0 <= tbl_end < len(pid)
    return pid[:tbl_end]


def to_row_pop_passages_a(train_table, all_row_ents, k_rows=3):
    """
    this version breaks tables into chunks of rows that contain at least one entity
    :param train_table:
    :param all_row_ents:
    :param k_rows:
    :return:
    """
    row_pops = []
    header = train_table['header']
    rows = train_table['rows']
    row_reps = []
    rstart = 0
    for rndx in range(len(rows)):
        row, row_ents = rows[rndx], all_row_ents[rndx]
        if not any(row_ents):
            continue
        cell_strs = [cell if not ent else f'«{cell}»' for cell, ent in zip(row, row_ents)]
        row_rep = ' * '.join([f'{header[cndx]}: {cell_strs[cndx]}' for cndx in range(len(row))])
        row_reps.append(row_rep)
        if len(row_reps) >= k_rows:
            pid = train_table['table_id']+f':R[{rstart}-{rndx}]'
            row_pops.append({'pid': pid, 'title': train_table['title'], 'text': ' | '.join(row_reps)})
            row_reps = []
            rstart = rndx+1
    if len(row_reps) > 0:
        pid = train_table['table_id'] + f':R[{rstart}-{len(rows)}]'
        row_pops.append({'pid': pid, 'title': train_table['title'], 'text': ' | '.join(row_reps)})
    return row_pops


def to_row_pop_passages_b(train_table, all_row_ents):
    row_pops = []
    header = train_table['header']
    rows = train_table['rows']
    for cndx in range(len(header)):
        all_col_ents = [rows[rndx][cndx] for rndx in range(len(rows)) if all_row_ents[rndx][cndx]]
        if len(all_col_ents) > 0:
            pid = train_table['table_id'] + f':C{cndx}'
            title = train_table['title'] + ' * ' + ' * '.join(header)
            text = f'{header[cndx]}: ' + ', '.join([f'«{cell}»' for cell in all_col_ents])
            # TODO: track max length of text in passages
            row_pops.append({'pid': pid, 'title': title, 'text': text})
    return row_pops


# TODO: try linearization in https://arxiv.org/pdf/2201.05966.pdf
# def to_row_pop_passages_tapex(train_table, all_row_ents):

def to_col_pop_passages(train_table, k_examples=2):
    header = train_table['header']
    rows = train_table['rows']
    examples = [', '.join([rows[rndx][cndx] for rndx in range(k_examples)]) for cndx in range(len(header))]
    header_rep = ' * '.join([f'«{h}»: {ex}' for h, ex in zip(header, examples)])
    return [{'pid': train_table['table_id'], 'title': train_table['title'], 'text': header_rep}]


def main(opts: Options):
    test_tables = dict()  # id2split
    for file in os.listdir(opts.split_definitions):
        with open(os.path.join(opts.split_definitions, file)) as f:
            test_table_list = json.load(f)
            for table_id in test_table_list:
                test_tables[table_id] = file[:-5]

    test_splits = list(set(test_tables.values()))
    test_splits.sort()
    # CONSIDER: track the max cell length in test, max title length - use it to filter train tables
    truncated_title_count = 0
    max_test_title_len = 0

    assert test_splits == ['column_id_test',  'column_id_validation', 'row_id_test',  'row_id_validation']
    split2out = {split_name: write_open(os.path.join(opts.query_dir, f'{split_name}.jsonl'))
                 for split_name in test_splits + ['train']}
    with write_open(os.path.join(opts.passage_dir, 'row', 'a.jsonl.gz')) as row_pop_a, \
            write_open(os.path.join(opts.passage_dir, 'row', 'b.jsonl.gz')) as row_pop_b, \
            write_open(os.path.join(opts.passage_dir, 'col', 'a.jsonl.gz')) as col_pop:
        for file in os.listdir(opts.table_dir):
            with open(os.path.join(opts.table_dir, file)) as f:
                tables = json.load(f)
            for table_id, table in tables.items():
                header_ents = [is_entity(h) for h in table['title']]
                header = [cell_text(h, opts.max_char_cell) for h in table['title']]
                full_title = table['pgTitle'] + ' ' + table['caption']
                title = truncate(full_title, opts.max_char_title)
                rows = [[cell_text(cell, opts.max_char_cell) for cell in row] for row in table['data']]
                row_ents = [is_entity(row[0]) for row in table['data']] \
                    if len(table['data']) > 0 and len(table['data'][0]) > 0 else []
                all_row_ents = [[is_entity(c) for c in row] for row in table['data']] \
                    if len(table['data']) > 0 and len(table['data'][0]) > 0 else []
                assert all([len(row) == len(header) for row in rows])

                if len(header) < 4:
                    assert table_id not in test_tables
                    continue
                if len(rows) < 4:
                    assert table_id not in test_tables
                    continue
                if any([h == header[0] for h in header[1:]]) and table_id not in test_tables:
                    # a few such tables in test
                    continue

                if len(full_title) > opts.max_char_title:
                    truncated_title_count += 1

                if table_id in test_tables:
                    max_test_title_len = max(max_test_title_len, len(full_title))
                    split = test_tables[table_id]
                    qout = split2out[split]
                    test_table = {'table_id': table_id, 'title': title, 'header': header, 'rows': rows,
                                          'row_entities': row_ents, 'header_entities': header_ents}
                    qout.write(json.dumps(test_table)+'\n')
                    continue

                # write out train queries
                train_table = {'table_id': table_id, 'title': title, 'header': header, 'rows': rows,
                                           'row_entities': row_ents, 'header_entities': header_ents}
                qout = split2out['train']
                qout.write(json.dumps(train_table) + '\n')

                # format passages separately for row-population and col-population
                for row_pop_passage in to_row_pop_passages_a(train_table, all_row_ents):
                    row_pop_a.write(json.dumps(row_pop_passage)+'\n')
                for row_pop_passage in to_row_pop_passages_b(train_table, all_row_ents):
                    row_pop_b.write(json.dumps(row_pop_passage)+'\n')
                for col_pop_passage in to_col_pop_passages(train_table):
                    col_pop.write(json.dumps(col_pop_passage)+'\n')

    for out in split2out.values():
        out.close()
    print(f'Max test title length = {max_test_title_len}')
    print(f'Truncated title count = {truncated_title_count}')


if __name__ == "__main__":
    opts = Options()
    fill_from_args(opts)
    main(opts)
