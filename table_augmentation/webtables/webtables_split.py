from util.line_corpus import jsonl_lines, write_open, block_shuffle, shuffled_writer
from util.args_help import fill_from_args
import os
import random
from collections import Counter
import ujson as json
import unicodedata
import numpy as np


class Options:
    def __init__(self):
        self.tables = ''
        self.output_dir = ''
        self.dev_count = 10000
        self.test_count = 10000
        self.train_count = 1000000
        self.min_entity_count = 3
        self._required_args = ['tables', 'output_dir']


def normalize(text: str):
    return unicodedata.normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode('ascii').strip()


def entity_counts_from_file(filename: str):
    entity_counts = Counter()
    for line in jsonl_lines(filename):
        jobj = json.loads(line)
        key_col_ndx = jobj['key_col_ndx']
        if key_col_ndx < 0:
            continue
        key_col = [row[key_col_ndx] for row in jobj['rows']]
        for ent in key_col:
            entity_counts[normalize(ent)] += 1
    return entity_counts


def mark_candidates(jobj, entity_counts):
    candidate_cells = np.zeros((len(jobj['rows']), len(jobj['header'])), dtype=np.int8)
    # set the candidate cells to the entities above min_entity_count
    for rndx, row in enumerate(jobj['rows']):
        for cndx, cell in enumerate(row):
            ent = normalize(cell)
            if len(ent) > 0 and ent in entity_counts and entity_counts[ent] >= opts.min_entity_count:
                candidate_cells[rndx, cndx] = 1
    jobj['candidate_cells'] = candidate_cells.tolist()


def mark_candidates_test(line: str) -> str:
    jobj = json.loads(line)
    candidate_cells = np.zeros((len(jobj['rows']), len(jobj['header'])), dtype=np.int8)
    candidate_cells[:, jobj['key_col_ndx']] = 1
    jobj['candidate_cells'] = candidate_cells.tolist()
    return json.dumps(jobj) + '\n'


if __name__ == '__main__':
    opts = Options()
    fill_from_args(opts)
    dev_count = 0
    test_count = 0
    train_count = 0
    sample_prob = 0.5
    rand = random.Random(1234)
    total_count = opts.train_count + opts.dev_count + opts.test_count
    total_count *= 1/sample_prob
    with write_open(os.path.join(opts.output_dir, 'tmp_train.jsonl.gz')) as train_out, \
        write_open(os.path.join(opts.output_dir, 'dev.jsonl.gz')) as dev_out, \
        write_open(os.path.join(opts.output_dir, 'test.jsonl.gz')) as test_out:
        for line in block_shuffle(jsonl_lines(opts.tables, shuffled=rand), rand=rand):
            r = rand.random()
            if r < opts.train_count/total_count and train_count < opts.train_count:
                train_out.write(line)
                train_count += 1
            elif r < (opts.train_count + opts.dev_count) and dev_count < opts.dev_count:
                dev_out.write(mark_candidates_test(line))
                dev_count += 1
            elif r < (opts.train_count + opts.dev_count + opts.test_count) and test_count < opts.test_count:
                test_out.write(mark_candidates_test(line))
                test_count += 1
            else:
                pass
    print(f'wrote {train_count} train, {dev_count} dev, {test_count} test')
    entity_counts = entity_counts_from_file(os.path.join(opts.output_dir, 'tmp_train.jsonl.gz'))
    with shuffled_writer(os.path.join(opts.output_dir, 'train')) as train_out:
        for line in jsonl_lines(os.path.join(opts.output_dir, 'tmp_train.jsonl.gz')):
            jobj = json.loads(line)
            mark_candidates(jobj, entity_counts)
            train_out.write(json.dumps(jobj)+'\n')
    os.remove(os.path.join(opts.output_dir, 'tmp_train.jsonl.gz'))
