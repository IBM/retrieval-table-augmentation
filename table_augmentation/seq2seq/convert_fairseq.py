import ujson as json
from util.line_corpus import jsonl_lines, write_open, block_shuffle
from util.args_help import fill_from_args
from table_augmentation.augmentation_tasks import TaskOptions, Table, Query
from typing import Callable, List
import os
import random


class Options:
    def __init__(self):
        self.train_tables = ''
        self.dev_tables = ''
        self.test_tables = ''
        self.out_dir = ''
        self.train_limit = -1
        self.is_query = False
        self.task = TaskOptions()


def convert(query_maker: Callable[[Table], List[Query]], tables: str, out_filename: str, limit: int, *,
            with_ids=False, sample_shuffle=True):
    if with_ids:
        fids = write_open(out_filename+'.ids')
    else:
        fids = None
    with write_open(out_filename+'.inp') as finp, write_open(out_filename+'.oup') as foup:
        qcount = 0
        lines = jsonl_lines(tables)
        if limit > 0 and sample_shuffle:
            lines = block_shuffle(lines, block_size=500000, rand=random.Random(123))
        for line in lines:
            jobj = json.loads(line)
            if query_maker is not None:
                table = Table.from_dict(jobj)
                queries = query_maker(table)
            else:
                queries = [Query.from_dict(jobj)]
            for query in queries:
                source = query.title + '. ' + query.text
                target = '; '.join(a.replace('; ', ', ') for a in query.answers)
                finp.write(source.replace('\n', ' ').strip()+'\n')
                foup.write(target.replace('\n', ' ').strip()+'\n')
                if fids is not None:
                    assert '\n' not in query.qid
                    fids.write(query.qid+'\n')
                qcount += 1
                if 0 < limit <= qcount:
                    if fids is not None:
                        fids.close()
                    return
    if fids is not None:
        fids.close()


if __name__ == "__main__":
    opts = Options()
    fill_from_args(opts)
    query_maker = opts.task.get_query_maker() if not opts.is_query else None
    if opts.train_tables:
        convert(query_maker, opts.train_tables, str(os.path.join(opts.out_dir, 'train')), opts.train_limit)
    if opts.dev_tables:
        convert(query_maker, opts.dev_tables, str(os.path.join(opts.out_dir, 'dev')), -1, with_ids=True)
    if opts.test_tables:
        convert(query_maker, opts.test_tables, str(os.path.join(opts.out_dir, 'test')), -1, with_ids=True)
