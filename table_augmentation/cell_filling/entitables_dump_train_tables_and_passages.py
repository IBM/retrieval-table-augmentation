import logging
import os

import tqdm
import ujson as json

from table_augmentation.cell_filling.utils import HOME, process_table, fill_from_args_, to_passages_a
from util.line_corpus import write_open

logger = logging.getLogger(__name__)


class Options:
    def __init__(self):
        # inputs
        self.table_dir = os.path.join(HOME, "data/WP_table/tables_redi2_1")
        self.test_query = os.path.join(HOME, "codes/cikm2019-table/data/qrel/gt_all_norm.tsv")
        # outputs
        self.output_dir = os.path.join(HOME, "data/cell_filling/dpr")
        # filtering options
        self.max_char_title = 100
        self.max_char_cell = 50


def main(opts: Options):
    test_tables = set()  # id2split
    with open(opts.test_query) as f:
        for line in f:
            table_id = line.strip().split()[0]
            table_id = "-".join(table_id.split("-")[:3])  # tables, table_id, table_segment, row
            test_tables.add(table_id)

    # CONSIDER: track the max cell length in test, max title length - use it to filter train tables
    truncated_title_count = 0
    total_title_count = 0

    with write_open(os.path.join(opts.output_dir, 'passages/a.jsonl.gz')) as out_p, \
            write_open(os.path.join(opts.output_dir, 'queries/all_tables.jsonl.gz')) as out_q:  # NOTE: nontest_tables.jsonl.gz
        pbar = tqdm.tqdm(os.listdir(opts.table_dir))
        for file in pbar:
            with open(os.path.join(opts.table_dir, file)) as f:
                tables = json.load(f)

            for table_id, table in tables.items():
                train_table = process_table(table_id, table, opts)

                if train_table["table_id"] in test_tables:
                    continue

                if len(train_table['title']) > opts.max_char_title:
                    truncated_title_count += 1
                total_title_count += 1

                # format passages separately for row-population and col-population
                out_q.write(json.dumps(train_table) + '\n')
                for passage in to_passages_a(train_table):
                    out_p.write(json.dumps(passage) + '\n')

    print(f'Truncated title count = {truncated_title_count} out of {total_title_count}')


if __name__ == "__main__":
    opts = Options()
    fill_from_args_(opts)
    main(opts)
