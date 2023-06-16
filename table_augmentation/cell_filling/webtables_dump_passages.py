import logging
import os

import tqdm
import ujson as json

from table_augmentation.cell_filling.utils import HOME, fill_from_args_, to_passages_a
from table_augmentation.normalization import NormalizationStyle
from util.line_corpus import write_open, jsonl_lines

logger = logging.getLogger(__name__)


class Options:
    def __init__(self):
        # inputs
        self.train_dir = ''
        # outputs
        self.output_dir = ''
        # filtering options
        self.max_char_title = 100
        self.max_char_cell = 50

        self.normalize = NormalizationStyle.identity
        self._required_args = ['train_dir', 'output_dir']


def main(opts: Options):
    # CONSIDER: track the max cell length in test, max title length - use it to filter train tables
    truncated_title_count = 0
    total_title_count = 0

    normalizer = opts.normalize.get_normalizer()

    with write_open(os.path.join(opts.output_dir, 'passages/a.jsonl.gz')) as out_p:
        for line in tqdm.tqdm(jsonl_lines(opts.train_dir), total=sum(1 for _ in jsonl_lines(opts.train_dir))):
            train_table = json.loads(line)

            if len(train_table['title']) > opts.max_char_title:
                truncated_title_count += 1
            total_title_count += 1

            for passage in to_passages_a(train_table, normalizer=normalizer):
                out_p.write(json.dumps(passage) + '\n')

    print(f'Truncated title count = {truncated_title_count} out of {total_title_count}')


if __name__ == "__main__":
    opts = Options()
    fill_from_args_(opts)
    main(opts)
