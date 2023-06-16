from table_augmentation.normalization import NormalizationStyle
from util.line_corpus import jsonl_lines, write_open, shuffled_writer
from util.args_help import fill_from_args
from table_augmentation.table import Table
from typing import Callable, List, Tuple
import os
import ujson as json
from collections import defaultdict
from util.reporting import Reporting
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


class Options:
    def __init__(self):
        self.orig_tables = ''
        self.deduplicated_dir = ''
        self.splits = 32
        self.title_unique = False
        self._required_args = ['orig_tables', 'deduplicated_dir']


_WHITESPACE = re.compile(r'\s+')


def table_normalized_string(table: Table, opts: Options) -> str:
    text = ','.join(table.header) + '\n' + '|'.join('*'.join(row) for row in table.rows)
    if opts.title_unique:
        text = table.title + text
    text = unicodedata.normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode('ascii').strip()
    text = _WHITESPACE.sub(' ', text)
    return text


logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
opts = Options()
fill_from_args(opts)
report = Reporting()

if not os.path.exists(os.path.join(opts.deduplicated_dir, 'tmp')):
    all_table_ids = set()
    tmps = [write_open(os.path.join(opts.deduplicated_dir, 'tmp', f'{i}.jsonl.gz')) for i in range(opts.splits)]
    large_prime = 27644437
    for line in jsonl_lines(opts.orig_tables):
        if report.is_time():
            logger.info(report.progress_str('initial table'))
        table = Table.from_dict(json.loads(line))
        assert table.table_id not in all_table_ids
        all_table_ids.add(table.table_id)
        table_str = table_normalized_string(table, opts)
        bucket = (hash(table_str) % large_prime) % opts.splits
        tmps[bucket].write(json.dumps({'table_id': table.table_id, 'table_string': table_str})+'\n')
    for tmp in tmps:
        tmp.close()

    logger.info('Wrote content grouped files')

duplicated_table_ids = []
for tmp_file in os.listdir(os.path.join(opts.deduplicated_dir, 'tmp')):
    dups = defaultdict(list)
    for line in jsonl_lines(os.path.join(opts.deduplicated_dir, 'tmp', tmp_file)):
        if report.is_time():
            logger.info(report.progress_str('content normalized tables'))
        jobj = json.loads(line)
        dups[jobj['table_string']].append(jobj['table_id'])
    for table_ids in dups.values():
        if len(table_ids) > 1:
            duplicated_table_ids.extend(table_ids[1:])

with write_open(os.path.join(opts.deduplicated_dir, 'dup_table_ids.jsonl')) as out:
    for dt in duplicated_table_ids:
        out.write(json.dumps({'table_id': dt})+'\n')

duplicated_table_ids = set(duplicated_table_ids)
with shuffled_writer(os.path.join(opts.deduplicated_dir, 'deduped')) as out:
    for line in jsonl_lines(opts.orig_tables):
        if report.is_time():
            logger.info(report.progress_str('deduplicated tables'))
        table = Table.from_dict(json.loads(line))
        if table.table_id not in duplicated_table_ids:
            out.write(line)
