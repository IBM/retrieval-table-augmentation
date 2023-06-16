from collections import Counter
from util.line_corpus import jsonl_lines, jsonl_files, write_open, shuffled_writer, block_shuffle
from util.args_help import fill_from_args
import ujson as json
import unicodedata
import multiprocessing
from multiprocessing.pool import ThreadPool
import functools
import logging
import numpy as np
import tabulate
import os
from typing import List, Dict, Optional
import random

logger = logging.getLogger(__name__)


class Options:
    def __init__(self):
        self.tables = ''
        self.filtered_tables = ''
        self.num_threads = multiprocessing.cpu_count()
        self.header_count_file = ''
        self.allow_non_key_col = False
        self.entity_count_file = ''
        self.display_header_counts = False
        self.display_entity_counts = False
        self.min_header_count = 10
        self.fraction_headers_above_min = 0.8
        self.min_entity_count = 3
        self.fraction_entities_above_min = 0.8
        self.dev_count = 10000
        self.test_count = 10000
        self._required_args = ['tables', 'header_count_file', 'entity_count_file']


def normalize(text: str):
    return unicodedata.normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode('ascii').strip()


def header_counts_from_file(filename, opts: Options):
    header_counts = Counter()
    for line in jsonl_lines(filename):
        jobj = json.loads(line)
        if not opts.allow_non_key_col and jobj['key_col_ndx'] < 0:
            continue
        for h in jobj['header']:
            nh = normalize(h)
            if len(nh) > 0:
                # CONSIDER: require at least one word char?
                header_counts[nh] += 1
    return header_counts


def write_header_file(pool: ThreadPool, opts: Options):
    filenames = jsonl_files(opts.tables)
    header_counts_from_file_c = functools.partial(header_counts_from_file, opts=opts)
    header_counts_list = pool.map(header_counts_from_file_c, filenames)
    header_counts = header_counts_list[0]
    for hc in header_counts_list[1:]:
        for h, c in hc.items():
            header_counts[h] += c

    with write_open(opts.header_count_file) as out:
        for h, c in header_counts.items():
            if c > 1:
                out.write(json.dumps({h: c})+'\n')


def table_counts_from_file(filename,
                           header_counts: Dict[str, int],
                           min_counts: List[int], header_fractions: List[float],
                           opts: Options):
    table_counts = np.zeros((len(min_counts), len(header_fractions)), dtype=np.int32)
    up_to_min_freq = np.zeros(len(min_counts))
    for line in jsonl_lines(filename):
        jobj = json.loads(line)
        if not opts.allow_non_key_col and jobj['key_col_ndx'] < 0:
            continue
        up_to_min_freq[:] = 0
        for h in jobj['header']:
            nh = normalize(h)
            if len(nh) > 0 and nh in header_counts:
                hc = header_counts[nh]
                for ndx, min in enumerate(min_counts):
                    if hc >= min:
                        up_to_min_freq[ndx] += 1
        up_to_min_freq /= len(jobj['header'])
        for min_ndx in range(len(min_counts)):
            for frac_ndx in range(len(header_fractions)):
                if up_to_min_freq[min_ndx] >= header_fractions[frac_ndx]:
                    table_counts[min_ndx, frac_ndx] += 1
    return table_counts


def tables_by_freq_header(pool: ThreadPool, header_counts: Dict[str, int], opts: Options):
    min_counts = [2, 3, 5, 10]
    header_fractions = [1.0, 0.9, 0.8]

    filenames = jsonl_files(opts.tables)
    table_counts_from_file_c = functools.partial(table_counts_from_file,
                                                 header_counts=header_counts,
                                                 min_counts=min_counts, header_fractions=header_fractions,
                                                 opts=opts)
    table_counts_list = pool.map(table_counts_from_file_c, filenames)

    table_counts = table_counts_list[0]  # len(min_counts) x len(header_fractions)
    for tc in table_counts_list[1:]:
        table_counts += tc
    display_header = ['Header Fraction'] + [f'Min Freq {mc}' for mc in min_counts]
    display_rows = [[header_fractions[frac_ndx]] + table_counts[:, frac_ndx].tolist()
                    for frac_ndx in range(len(header_fractions))]
    print(tabulate.tabulate(display_rows, headers=display_header))


def passes_header_filter(table, header_counts: Dict[str, int],
                         min_header_count: int, fraction_headers_above_min: float):
    headers_past_min_count = 0
    for h in table['header']:
        nh = normalize(h)
        if len(nh) > 0 and nh in header_counts and header_counts[nh] >= min_header_count:
            headers_past_min_count += 1
    return headers_past_min_count / len(table['header']) >= fraction_headers_above_min


def entity_counts_from_file(filename: str, header_counts: Dict[str, int], opts: Options):
    entity_counts = Counter()
    for line in jsonl_lines(filename):
        jobj = json.loads(line)
        key_col_ndx = jobj['key_col_ndx']
        if key_col_ndx < 0:
            continue
        if not passes_header_filter(jobj, header_counts, opts.min_header_count, opts.fraction_headers_above_min):
            continue
        key_col = [row[key_col_ndx] for row in jobj['rows']]
        for ent in key_col:
            entity_counts[normalize(ent)] += 1
    return entity_counts


def write_key_entity_counts(pool: ThreadPool, header_counts: Dict[str, int], opts: Options):
    # consider tables that pass our column filter
    # we will gather counts of all entities in the key column
    filenames = jsonl_files(opts.tables)
    entity_counts_from_file_c = functools.partial(entity_counts_from_file,
                                                 header_counts=header_counts,
                                                 opts=opts)
    entity_counts_list = pool.map(entity_counts_from_file_c, filenames)
    entity_counts = entity_counts_list[0]
    for ec in entity_counts_list[1:]:
        for e, c in ec.items():
            entity_counts[e] += c
    with write_open(opts.entity_count_file) as out:
        for e, c in entity_counts.items():
            if c > 1:
                out.write(json.dumps({e: c})+'\n')


def table_counts_filtered_from_file(filename,
                                    header_counts: Dict[str, int],
                                    entity_counts: Dict[str, int],
                                    min_counts: List[int], entity_fractions: List[float],
                                    opts: Options):
    table_counts = np.zeros((len(min_counts), len(entity_fractions)), dtype=np.int32)
    up_to_min_freq = np.zeros(len(min_counts))
    for line in jsonl_lines(filename):
        jobj = json.loads(line)
        key_col_ndx = jobj['key_col_ndx']
        if key_col_ndx < 0:
            continue
        if not passes_header_filter(jobj, header_counts, opts.min_header_count, opts.fraction_headers_above_min):
            continue
        up_to_min_freq[:] = 0
        ents = [normalize(row[key_col_ndx]) for row in jobj['rows']]
        for ent in ents:
            if len(ent) > 0 and ent in entity_counts:
                count = entity_counts[ent]
                for ndx, min in enumerate(min_counts):
                    if count >= min:
                        up_to_min_freq[ndx] += 1
        up_to_min_freq /= len(jobj['rows'])
        for min_ndx in range(len(min_counts)):
            for frac_ndx in range(len(entity_fractions)):
                if up_to_min_freq[min_ndx] >= entity_fractions[frac_ndx]:
                    table_counts[min_ndx, frac_ndx] += 1
    return table_counts


def tables_by_freq_header_and_entity(pool: ThreadPool,
                                     header_counts: Dict[str, int], entity_counts: Dict[str, int],
                                     opts: Options):
    min_counts = [2, 3, 5, 10]
    entity_fractions = [1.0, 0.9, 0.8]
    filenames = jsonl_files(opts.tables)
    table_counts_filtered_from_file_c = functools.partial(table_counts_filtered_from_file,
                                                 header_counts=header_counts,
                                                 entity_counts=entity_counts,
                                                 min_counts=min_counts, entity_fractions=entity_fractions,
                                                 opts=opts)
    table_counts_list = pool.map(table_counts_filtered_from_file_c, filenames)

    table_counts = table_counts_list[0]  # len(min_counts) x len(entity_fractions)
    for tc in table_counts_list[1:]:
        table_counts += tc
    display_header = ['Entity Fraction'] + [f'Min Freq {mc}' for mc in min_counts]
    display_rows = [[entity_fractions[frac_ndx]] + table_counts[:, frac_ndx].tolist()
                    for frac_ndx in range(len(entity_fractions))]
    print(tabulate.tabulate(display_rows, headers=display_header))


def filter_tables(header_counts: Dict[str, int], entity_counts: Dict[str, int], opts: Options):
    written = 0
    rand = random.Random(1234)
    test_count = 0
    dev_count = 0
    with shuffled_writer(os.path.join(opts.filtered_tables, 'train')) as train_out, \
            write_open(os.path.join(opts.filtered_tables, 'dev.jsonl.gz')) as dev_out, \
            write_open(os.path.join(opts.filtered_tables, 'test.jsonl.gz')) as test_out:
        for line in block_shuffle(jsonl_lines(opts.tables, shuffled=rand), rand=rand, block_size=100000):
            jobj = json.loads(line)
            key_col_ndx = jobj['key_col_ndx']
            if key_col_ndx < 0:
                continue
            if not passes_header_filter(jobj, header_counts, opts.min_header_count, opts.fraction_headers_above_min):
                continue
            up_to_min_freq = 0
            ents = [normalize(row[key_col_ndx]) for row in jobj['rows']]
            for ent in ents:
                if len(ent) > 0 and ent in entity_counts:
                    count = entity_counts[ent]
                    if count >= opts.min_entity_count:
                        up_to_min_freq += 1
            up_to_min_freq /= len(jobj['rows'])
            if up_to_min_freq >= opts.fraction_entities_above_min:
                candidate_cells = np.zeros((len(jobj['rows']), len(jobj['header'])), dtype=np.int8)
                # set the candidate cells to the entities above min_entity_count
                for rndx, row in enumerate(jobj['rows']):
                    for cndx, cell in enumerate(row):
                        ent = normalize(cell)
                        if len(ent) > 0 and ent in entity_counts and entity_counts[ent] >= opts.min_entity_count:
                            candidate_cells[rndx, cndx] = 1
                jobj['candidate_cells'] = candidate_cells.tolist()
                written += 1
                if test_count < opts.test_count and rand.random() < 0.02:
                    test_out.write(json.dumps(jobj)+'\n')
                    test_count += 1
                elif dev_count < opts.dev_count and rand.random() < 0.02:
                    dev_out.write(json.dumps(jobj)+'\n')
                    dev_count += 1
                else:
                    train_out.write(json.dumps(jobj)+'\n')
    print(f'Wrote {written} filtered tables; {test_count} in test, {dev_count} in dev')


def load_counts(filename: str, min_count:int = 1) -> Dict[str, int]:
    counts = dict()
    for line in jsonl_lines(filename):
        jobj = json.loads(line)
        h, c = next(iter(jobj.items()))
        if c > min_count:
            counts[h] = c
    return counts


if __name__ == '__main__':
    opts = Options()
    fill_from_args(opts)
    pool = ThreadPool(processes=opts.num_threads)
    print(f'Using multiprocessing pool with {opts.num_threads} workers')

    # find header counts
    if not os.path.exists(opts.header_count_file):
        write_header_file(pool, opts)

    # read header counts
    header_counts = load_counts(opts.header_count_file,
                                min_count=1 if opts.display_header_counts else opts.min_header_count)
    print(f'Loaded {len(header_counts)} header counts')

    # display how many tables we will have with various filters
    if opts.display_header_counts:
        tables_by_freq_header(pool, header_counts, opts)

    # find entity counts
    if not os.path.exists(opts.entity_count_file):
        write_key_entity_counts(pool, header_counts, opts)

    entity_counts = load_counts(opts.entity_count_file)
    print(f'Loaded {len(entity_counts)} key column entity counts')

    if opts.display_entity_counts:
        # display tables once filtered by both header and entity frequency
        tables_by_freq_header_and_entity(pool, header_counts, entity_counts, opts)

    if opts.filtered_tables:
        filter_tables(header_counts, entity_counts, opts)

