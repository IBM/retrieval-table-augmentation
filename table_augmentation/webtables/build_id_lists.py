from util.args_help import fill_from_args
from util.line_corpus import jsonl_lines, write_open
import ujson as json
import os


class Options:
    def __init__(self):
        self.webtables_dir = ''
        self.cell_filling_style = False


if __name__ == '__main__':
    opts = Options()
    fill_from_args(opts)

    for tables, id_list in [('train', 'train_ids.txt'), ('dev.jsonl.gz', 'dev_ids.txt'), ('test.jsonl.gz', 'test_ids.txt')]:
        with write_open(os.path.join(opts.webtables_dir, id_list)) as ids_out:
            for line in jsonl_lines(os.path.join(opts.webtables_dir, tables)):
                jobj = json.loads(line)
                if opts.cell_filling_style:
                    table_id = jobj['id']
                else:
                    table_id = jobj['table_id']
                assert '\n' not in table_id
                ids_out.write(table_id+'\n')
