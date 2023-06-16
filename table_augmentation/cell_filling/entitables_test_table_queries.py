import logging
import os

import ujson as json

from table_augmentation.cell_filling.utils import HOME, normalize_entity, process_table, fill_from_args_
from util.line_corpus import write_open

logger = logging.getLogger(__name__)


class Options:
    def __init__(self):
        # inputs
        self.table_dir = os.path.join(HOME, "data/WP_table/tables_redi2_1")
        self.test_query_gt = os.path.join(HOME, "codes/cikm2019-table/data/qrel/gt_all_norm.tsv")
        self.test_query_table = os.path.join(HOME, "codes/cikm2019-table/query/query_table.txt")
        # outputs
        self.output_path = os.path.join(HOME, "data/cell_filling/dpr/queries/test_tables.jsonl.gz")
        # filtering options
        self.max_char_title = 100
        self.max_char_cell = 50


def main(opts: Options):
    query_table = {}
    with open(opts.test_query_table) as f:
        for line in f:
            line = line.strip().split()
            query_table[line[0]] = [int(line[1]), int(line[2]), ]

    test_tables = []
    with open(opts.test_query_gt) as f:
        for line in f:
            table_row_id, _, answer, _ = line.strip().split()
            row_id = int(table_row_id.split("-")[-1])
            table_id = "-".join(table_row_id.split("-")[:-1])
            col_id = query_table[table_id][1]
            answer = normalize_entity(answer)
            if len(test_tables) == 0 or test_tables[-1]['table_id'] != table_id:
                test_tables.append(dict(table_id=table_id, answers=[]))
            if len(test_tables[-1]["answers"]) == 0 or test_tables[-1]["answers"][-1][0] != row_id:
                test_tables[-1]["answers"].append([row_id, col_id, [], ])
            test_tables[-1]["answers"][-1][-1].append(answer)
    test_tables = sorted(test_tables, key=lambda x: x['table_id'])

    # CONSIDER: track the max cell length in test, max title length - use it to filter train tables
    truncated_title_count = 0
    total_title_count = 0

    last_tables = None
    last_table_fn = None
    with write_open(opts.output_path) as out_q:
        for test_table in test_tables:
            table_id = test_table['table_id']
            table_fn = table_id.split("-")[1]
            if table_fn != last_table_fn:
                with open(os.path.join(opts.table_dir, "re_tables-{}.json".format(table_fn))) as f:
                    tables = json.load(f)
                last_table_fn = table_fn
                last_tables = tables
            else:
                tables = last_tables

            train_table = process_table(table_id, tables[table_id], opts)
            full_title = train_table["title"]
            if len(full_title) > opts.max_char_title:
                truncated_title_count += 1
            total_title_count += 1

            # format passages separately for row-population and col-population
            train_table['answers'] = test_table['answers']
            out_q.write(json.dumps(train_table) + '\n')

    print(f'Truncated title count = {truncated_title_count} out of {total_title_count}')


if __name__ == "__main__":
    opts = Options()
    fill_from_args_(opts)
    main(opts)
