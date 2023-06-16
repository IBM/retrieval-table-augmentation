from util.line_corpus import jsonl_lines, write_open
import ujson as json
from util.args_help import fill_from_args
from table_augmentation.augmentation_tasks import make_row_query
from table_augmentation.table import Table
from table_augmentation.normalization import NormalizationStyle
import os


class Options:
    def __init__(self):
        self.row_tables = ''
        self.tables = ''
        self.num_seeds = 2
        self.normalization = NormalizationStyle.deunicode


if __name__ == '__main__':
    opts = Options()
    fill_from_args(opts)
    normalizer = opts.normalization.get_normalizer()
    for split in ['train', 'dev.jsonl.gz', 'test.jsonl.gz']:
        with write_open(os.path.join(opts.row_tables, split if split.endswith('.jsonl.gz') else split+'.jsonl.gz')) as out:
            for line in jsonl_lines(os.path.join(opts.tables, split)):
                jobj = json.loads(line)
                table = Table.from_dict(jobj)
                queries = make_row_query(table, opts.num_seeds, normalizer)
                if queries is not None and len(queries) > 0:
                    out.write(line)
