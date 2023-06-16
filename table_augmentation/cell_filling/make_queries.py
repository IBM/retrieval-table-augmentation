import os

import tqdm
import ujson as json

from table_augmentation.augmentation_tasks import TaskOptions
from table_augmentation.cell_filling.bm25_dpr_training_data import (
    make_training_queries, DEFAULT_ENTITY_COLUMN_KEY, CellFilterStyle
)
from table_augmentation.cell_filling.utils import HOME
from util.args_help import fill_from_args
from util.line_corpus import jsonl_lines, write_open


class Options:
    def __init__(self):
        self.train_tables = os.path.join(HOME, "data/cell_filling/dpr/queries/all_tables.jsonl.gz")
        self.train_cell_tables = os.path.join(HOME, "data/cell_filling/reax/train.jsonl.gz")
        self.examples_per_table = 1
        self.chunk_size = 3
        self.entity_column_key = DEFAULT_ENTITY_COLUMN_KEY

        self.task = TaskOptions()
        self.cell_filter = CellFilterStyle.entitables


if __name__ == '__main__':
    opts = Options()
    fill_from_args(opts)

    total = 0
    for _ in jsonl_lines(opts.train_tables):
        total += 1

    answer_normalizer = opts.task.answer_normalization.get_normalizer()

    with write_open(opts.train_cell_tables) as out:
        for line in tqdm.tqdm(jsonl_lines(opts.train_tables), total=total):
            jobj = json.loads(line)
            for query in make_training_queries(opts, jobj, answer_normalizer):
                query['id'] = query['id'].replace("-[", "::[")
                out.write(json.dumps(query) + '\n')
