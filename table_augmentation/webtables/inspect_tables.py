from util.args_help import fill_from_args
from util.line_corpus import jsonl_lines, block_shuffle
import ujson as json
import tabulate
from table_augmentation.table import Table


class Options:
    def __init__(self):
        self.tables = ''
        self.__required_args__ = ['tables']


if __name__ == '__main__':
    opts = Options()

    fill_from_args(opts)
    for line in block_shuffle(jsonl_lines(opts.tables, shuffled=True)):
        jobj = json.loads(line)
        tbl = Table.from_dict(jobj)
        rows = [['«'+cell+'»' if tbl.candidate_cells[row_ndx, col_ndx] else cell for col_ndx, cell in enumerate(row)]
                for row_ndx, row in enumerate(tbl.rows)]
        print('ID: '+tbl.table_id)
        print('Title: '+tbl.title)
        print(tabulate.tabulate(rows, headers=tbl.header))
        k = input('...')


"""
python ${PYTHONPATH}/table_augmentation/webtables/inspect_tables.py \
--tables /data/webtables2015_en_relational/webtables_clean
"""