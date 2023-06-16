from util.line_corpus import jsonl_lines, write_open
import ujson as json
from util.args_help import fill_from_args
from util.reporting import Distribution
from table_augmentation.augmentation_tasks import make_row_query
from table_augmentation.table import Table
from table_augmentation.normalization import NormalizationStyle


class Options:
    def __init__(self):
        self.train_row_tables = ''
        self.train_tables = ''
        self.num_seeds = 2
        self.min_fraction_ents = 0.3
        self.normalization = NormalizationStyle.identity


if __name__ == '__main__':
    opts = Options()
    fill_from_args(opts)
    distribution = Distribution([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], num_samples_per_threshold=0)
    normalizer = opts.normalization.get_normalizer()
    with write_open(opts.train_row_tables) as out:
        for line in jsonl_lines(opts.train_tables):
            jobj = json.loads(line)
            table = Table.from_dict(jobj)
            num_ents = table.candidate_cells[opts.num_seeds:, table.key_col_ndx].sum()
            fraction_ents = num_ents / (len(table.rows) - opts.num_seeds)
            distribution.note_value(fraction_ents)
            if fraction_ents >= opts.min_fraction_ents:
                query = make_row_query(table, opts.num_seeds, normalizer)
            else:
                query = None

            if query is not None:
                out.write(line)
    distribution.display(show_counts=True, show_sums=False)