from util.line_corpus import jsonl_lines, write_open
from util.args_help import fill_from_args
from table_augmentation.augmentation_tasks import TaskOptions, Table
from multiprocessing.pool import ThreadPool
import multiprocessing
import functools
import os
import ujson as json
from typing import Callable, Tuple


class Options:
    def __init__(self):
        self.task = TaskOptions()
        self.train_tables = ''
        self.min_col = 3
        self.min_row = 3
        self.passages = ''
        self.num_threads = multiprocessing.cpu_count()
        self._required_args = ['train_tables', 'passages', 'task.task']


def table_file_to_passage_file(filename: str, opts: Options) -> Tuple[int, int]:
    passage_maker = opts.task.get_passage_maker()
    wrote_count = 0
    skip_count = 0
    with write_open(os.path.join(opts.passages, filename)) as out:
        for line in jsonl_lines(os.path.join(opts.train_tables, filename)):
            jobj = json.loads(line)
            table = Table.from_dict(jobj)
            if len(table.header) < opts.min_col or len(table.rows) < opts.min_row:
                skip_count += 1
                continue
            for passage in passage_maker(table):
                out.write(json.dumps(passage.to_dict())+'\n')
                wrote_count += 1
    return wrote_count, skip_count


if __name__ == '__main__':
    opts = Options()
    fill_from_args(opts)

    pool = ThreadPool(processes=opts.num_threads)
    print(f'Using multiprocessing pool with {opts.num_threads} workers')

    table_file_to_passage_file_f: Callable[[str], Tuple[int, int]] = \
        functools.partial(table_file_to_passage_file, opts=opts)

    wrote_count = pool.map(table_file_to_passage_file_f, os.listdir(opts.train_tables))
    pool.close()
    print(f'Wrote {[w[0] for w in wrote_count]} passages')
    print(f'Skipped {[w[1] for w in wrote_count]} tables')
