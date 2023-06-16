from util.line_corpus import write_open
from util.args_help import fill_from_args
import ujson as json
import os
import logging
from table_augmentation.table import Table
from table_augmentation.augmentation_tasks import make_col_passages, make_row_passages

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


def main(opts: Options):
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
    with write_open(os.path.join(opts.passage_dir, 'row', 'a.jsonl.gz')) as row_pop, \
            write_open(os.path.join(opts.passage_dir, 'col', 'a.jsonl.gz')) as col_pop:
        for file in os.listdir(opts.table_dir):
            with open(os.path.join(opts.table_dir, file)) as f:
                tables = json.load(f)
            for table_id, table_jobj in tables.items():
                table = Table.from_entitable(table_id, table_jobj,
                                             max_char_cell=opts.max_char_cell,
                                             max_char_title=opts.max_char_title)
                table.validate()

                if len(table.header) < 4 or len(table.rows) < 4:
                    assert table_id not in test_tables
                    continue

                # additional filter for train tables
                if any([h == table.header[0] for h in table.header[1:]]) and table_id not in test_tables:
                    # a few such tables in test
                    continue

                if table_id in test_tables:
                    split = test_tables[table_id]
                    qout = split2out[split]
                    qout.write(json.dumps(table.to_dict())+'\n')
                else:
                    # write out train tables
                    qout = split2out['train']
                    qout.write(json.dumps(table.to_dict()) + '\n')

                    # format passages separately for row-population and col-population
                    for row_pop_passage in make_row_passages(table, k_rows=3):
                        row_pop.write(json.dumps(row_pop_passage.to_dict())+'\n')
                    for col_pop_passage in make_col_passages(table, k_rows=2):
                        col_pop.write(json.dumps(col_pop_passage.to_dict())+'\n')

    for out in split2out.values():
        out.close()


if __name__ == "__main__":
    opts = Options()
    fill_from_args(opts)
    main(opts)
