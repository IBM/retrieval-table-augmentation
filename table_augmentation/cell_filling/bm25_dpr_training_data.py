import copy
import functools
import logging
import multiprocessing
import os
import random
from enum import Enum
from multiprocessing.pool import ThreadPool
from typing import Callable, List

import jnius_config
import tqdm
import ujson as json
from tabulate import tabulate
from termcolor import colored

from table_augmentation.augmentation_tasks import TaskOptions
from table_augmentation.cell_filling.utils import HOME, fill_from_args_, get_type
from table_augmentation.table_dpr_bm25_answer_bearing import is_answer_bearing
from util.line_corpus import jsonl_lines, shuffled_writer
from util.reporting import Reporting

logger = logging.getLogger(__name__)

DEFAULT_ENTITY_COLUMN_KEY = "entity_column"


class CellFilterStyle(Enum):
    entitables = 1
    webtables = 2
    webtables_string_only = 3

    def get_filter(self) -> Callable[[List[List[str]], int, int], bool]:
        def filter_entitables(rows, row_id, col_id):
            return rows[row_id][col_id] != "-"

        def filter_webtables(rows, row_id, col_id):
            x = rows[row_id][col_id]
            return len(x) > 5

        def filter_webtables_string_only(rows, row_id, col_id):
            return get_type(rows[row_id][col_id]) == 'string'

        if self == CellFilterStyle.entitables:
            return filter_entitables
        elif self == CellFilterStyle.webtables:
            return filter_webtables
        elif self == CellFilterStyle.webtables_string_only:
            return filter_webtables_string_only


class Options:
    def __init__(self):
        self.debug = False
        self.debug_shuffle = False
        self.num_processes = -1
        self.batch_size = -1

        self.max_answer_expand = -1
        self.num_hits = 50
        # self.max_negatives = 5  # number of negatives to gather for each positive
        # self.max_positives = 10  # create at most this many training instances per original instance (ignoring the other positives)
        self.jar = os.path.join(HOME, "codes/anserini/target/anserini-0.4.1-SNAPSHOT-fatjar.jar")
        self.anserini_index = os.path.join(HOME, "data/cell_filling/dpr/passages/anserini_passages_a/index/")
        self.num_files = 16
        self.chunk_size = 3

        self.keep_empty = False
        self.no_strict_validation = False
        self.exclude_tables = ""

        self.train_file = os.path.join(HOME, "data/cell_filling/dpr/queries/all_tables.jsonl.gz")
        self.output_dir = os.path.join(HOME, "data/cell_filling/dpr/queries/training_data")
        self.examples_per_table = 1
        self.entity_column_key = DEFAULT_ENTITY_COLUMN_KEY  # 'key_col_ndx' for webtables

        self.task = TaskOptions()
        self.cell_filter = CellFilterStyle.entitables

    def _post_argparse(self):
        if self.batch_size < 0:
            self.batch_size = 5 if self.debug else 1024
        if self.num_processes < 0:
            self.num_processes = 0 if self.debug else multiprocessing.cpu_count()
        if self.debug_shuffle:
            assert self.debug


class PseudoPool:
    def map(self, func, iterable, chunksize=None):
        return [func(x) for x in iterable]


def make_training_queries(opts: Options, jobj, answer_normalizer):
    jobj = copy.deepcopy(jobj)

    header = jobj['header']
    if opts.entity_column_key != DEFAULT_ENTITY_COLUMN_KEY:
        jobj[DEFAULT_ENTITY_COLUMN_KEY] = jobj.pop(opts.entity_column_key)
    entity_col = jobj[DEFAULT_ENTITY_COLUMN_KEY]
    if len(jobj['rows']) < 3 or len(set(header)) < 3 or entity_col < 0:  # filtering
        return []
    headers_to_use = [i for i in range(len(header)) if header[i] != header[entity_col]]
    if len(headers_to_use) == 0:  # no headers to use: almost impossible
        return []

    rows = jobj['rows']
    row_inds = []
    for s in range(0, len(jobj["rows"]) - opts.chunk_size + 1, opts.chunk_size):
        row_inds.append((s, s + opts.chunk_size))
    random.shuffle(row_inds)

    ret = []
    for s, e in row_inds[:opts.examples_per_table]:
        headers_to_use_ = [i for i in headers_to_use if rows[s][i] != rows[s][entity_col] and
                           opts.cell_filter.get_filter()(rows, s, i)]
        if len(headers_to_use_) == 0:
            continue
        i_header = random.choice(headers_to_use_)
        ret.append(make_query(jobj, s, i_header, range(s + 1, e)))
        ret[-1]["id"] = "{}-[{}-{}]-{}".format(jobj["table_id"], s, e, i_header)
        ret[-1]["answers"] = [answer_normalizer(ans) for ans in ret[-1]["answers"]]
    return ret


class BM25forDPR:
    def __init__(self, jar: str, index: str, opts: Options):
        jnius_config.set_classpath(jar)
        from jnius import autoclass
        self.JString = autoclass('java.lang.String')
        JSearcher = autoclass('io.anserini.search.SimpleSearcher')
        self.searcher = JSearcher(self.JString(index))
        self.opts = opts

        # NOTE: only thread-based pooling works with the JSearcher
        self.pool = ThreadPool(processes=opts.num_processes) if opts.num_processes > 0 else PseudoPool()
        logger.info(f'Using multiprocessing pool with {opts.num_processes} workers')
        self.no_negative_skip_count = 0
        self.no_positive_skip_count = 0
        self.answer_normalizer = opts.task.answer_normalization.get_normalizer()
        self.query_maker = opts.task.get_query_maker()
        self._retrieve_one = functools.partial(_retrieve_one, searcher=self.searcher, JString=self.JString,
                                               num_hits=opts.num_hits, lowercase=False,
                                               strict=not opts.no_strict_validation,
                                               answer_normalizer=self.answer_normalizer)
        self.written = 0

    def _write_batch(self, out, query_tuples, passages):
        for query_tuple, passage_pos_neg in zip(query_tuples, passages):
            inst_id = query_tuple['id']
            query_title = query_tuple['title']
            query_text = query_tuple['text']
            answers = query_tuple['answers']

            positive_passages, negative_passages = passage_pos_neg
            if len(negative_passages) == 0:
                if self.no_negative_skip_count == 0:
                    logger.warning(f'No negatives for "{query_title}\n\n{query_text}"\n   Answers: {answers}')
                self.no_negative_skip_count += 1
                if not self.opts.keep_empty:
                    continue
            elif len(positive_passages) == 0:
                if self.no_positive_skip_count == 0:
                    logger.warning(f'No positive for "{query_title}\n\n{query_text}"\n   Answers: {answers}')
                self.no_positive_skip_count += 1
                if not self.opts.keep_empty:
                    continue

            assert isinstance(positive_passages, list)
            # positive_passages = positive_passages[:self.opts.max_positives]
            # negative_passages = negative_passages[:self.opts.max_negatives]
            if self.opts.debug:
                print("\n" + "=" * 5 + " QUERY " + "=" * 5 + ' ' + query_title)
                table = copy.deepcopy(query_tuple['table']['rows'])
                table[query_tuple['row_id']][query_tuple['col_id']] = colored(answers[0], "red")
                q_header = query_tuple['table']['header']
                q_header[query_tuple['table']['entity_column']] = colored(
                    q_header[query_tuple['table']['entity_column']], 'cyan'
                )
                print(tabulate(table, headers=q_header))
                for i, p in enumerate(positive_passages):
                    print("\n" + "=" * 5 + " POS {:d} ".format(i + 1) + "=" * 5 + ' ' + p['title'])
                    try:
                        print_retrieved_table(p['text'].replace(
                            "«{}»".format(answers[0]), "«{}»".format(colored(answers[0], "red"))
                        ))
                    except:
                        print("\nSadly... table format issue.")
                        print(p['text'])
                        print()
                # for i, n in enumerate(negative_passages):
                #     print("\n" + "=" * 5 + " NEG {:d} ".format(i + 1) + "=" * 5 + ' ' + n['title'])
                #     print_retrieved_table(n['text'])
                input()
            else:
                # we have a single unified list of positive passages
                ret = {'id': inst_id, 'query': {'title': query_title, 'text': query_text}, 'answers': answers,
                       'positives': positive_passages, 'negatives': negative_passages}
                for k, v in query_tuple.items():
                    if 'answers' in k and k != 'answers':
                        ret[k] = v
                out.write(json.dumps(ret) + '\n')
            self.written += 1

    def create(self, inp_file, output_dir):
        excluded_ids = set()
        if self.opts.exclude_tables != "":
            for line in jsonl_lines(self.opts.exclude_tables):
                id_ = json.loads(line)['id']
                id_ = "-".join(id_.split("-")[:3])
                excluded_ids.add(id_)

        report = Reporting()
        with shuffled_writer(output_dir, num_files=self.opts.num_files) as out:
            query_tuples = []
            for i_line, line in tqdm.tqdm(enumerate(jsonl_lines(inp_file, shuffled=self.opts.debug_shuffle)),
                                          total=sum(1 for _ in jsonl_lines(inp_file))):
                jobj = json.loads(line)
                if jobj['table_id'] in excluded_ids:
                    continue

                for query in make_training_queries(self.opts, jobj, self.answer_normalizer):
                    query_tuples.append(query)
                    if len(query_tuples) >= self.opts.batch_size:
                        passages = self.pool.map(self._retrieve_one, query_tuples)
                        self._write_batch(out, query_tuples, passages)
                        query_tuples = []
                        if report.is_time():
                            instance_count = report.check_count * self.opts.batch_size
                            logger.info(f'On instance {instance_count}, '
                                        f'{instance_count / report.elapsed_seconds()} instances per second')
                            logger.info(f"{i_line + 1} lines have been processed")
                            if self.no_negative_skip_count > 0:
                                logger.info(f'{self.no_negative_skip_count} skipped for lack of negatives')
                            if self.no_positive_skip_count > 0:
                                logger.info(f'{self.no_positive_skip_count} skipped for lack of positives')

            if len(query_tuples) > 0:
                passages = self.pool.map(self._retrieve_one, query_tuples)
                self._write_batch(out, query_tuples, passages)
            instance_count = report.check_count * self.opts.batch_size
            logger.info(f'Finished {instance_count} instances; wrote {self.written} training triples. '
                        f'{instance_count / report.elapsed_seconds()} instances per second')
            if self.no_negative_skip_count > 0:
                logger.info(f'{self.no_negative_skip_count} skipped for lack of negatives')
            if self.no_positive_skip_count > 0:
                logger.info(f'{self.no_positive_skip_count} skipped for lack of positives')


def make_query(table, query_row_id, query_col_id, context_rows):
    row_ids = [query_row_id, ] + list(context_rows)
    assert len(set(row_ids)) == len(row_ids)

    table = {
        k: [v[i] for i in row_ids] if k in ["rows", "row_entities", ] and isinstance(v, list) else v
        for k, v in table.items()
    }
    table["rows"] = copy.deepcopy(table["rows"])
    answers = [table["rows"][0][query_col_id], ]
    table["rows"][0][query_col_id] = "[MASK]"

    # text = ' * '.join([h + ': ' + ', '.join([row[ndx] for row in rows]) for ndx, h in enumerate(header)])
    # NOTE: use row population's format. Makes more sense
    text = " | ".join([" * ".join([h + ': ' + c for h, c in zip(table["header"], row)]) for row in table["rows"]])
    return {
        'title': table["title"], 'text': text, 'answers': answers, 'table': table, 'row_id': 0, 'col_id': query_col_id,
    }


def validate_positive(query, answers, retrieved_title, retrieved_content, strict=True):
    assert len(answers) > 0
    answers = ['«' + ans + '»' for ans in answers]
    if all([ans not in retrieved_content for ans in answers]):
        return False

    if not strict:
        return True  # no further validation

    if query['col_id'] == query['table']['entity_column']:
        col_id = 0 if query['col_id'] > 0 else 1
    else:
        col_id = query['table']['entity_column']
    query_ent = query['table']['rows'][query['row_id']][col_id]
    if query_ent in retrieved_title:
        return True

    for row in retrieved_content.split(" | "):
        if any([ans in row for ans in answers]) and query_ent in row:
            return True
    return False


def _retrieve_one(query, searcher, JString, num_hits: int, lowercase: bool, strict: bool, answer_normalizer):
    query_str = query['title'] + '\n\n' + query['text']
    if len(query_str.split()) > 1024:  # I'll suggest directly drop everything: it's ridiculous
        return [], []
    hits = searcher.search(JString(query_str.encode('utf-8')), num_hits)
    pos = []
    negs = []
    for hit in hits:
        pid = hit.docid
        if pid.startswith(query['table']['table_id']):
            continue
        title = hit.content[:hit.content.find('\n\n')]
        text = hit.content[hit.content.find('\n\n') + 2:]

        assert not strict
        answer_bearing = is_answer_bearing(text, query['answers'], answer_normalizer)
        if answer_bearing:
            pos.append({'pid': pid, 'title': title, 'text': text})
        else:
            negs.append({'pid': pid, 'title': title, 'text': text})
    return pos, negs


def print_retrieved_table(q, no_special_token=False):
    header = None
    table = []
    for row in q.split(" | "):
        header_ = []
        table.append([])
        cells = row.split(" * ")
        for cell in cells:
            if no_special_token and cell.count(":") == 1:
                h, e = cell.split(":")
            elif "«" in cell:
                h, e = cell.split("«")
                assert h.strip().endswith(":")
                h = h.strip()[:-1]
                e = e.replace("»", "").strip()
            elif cell.strip().endswith(":"):
                h = cell.strip()[:-1]
                e = ""
            else:
                table[-1] += " * " + cell.strip()
                continue
            h = h.strip()
            e = e.strip()
            header_.append(h)
            table[-1].append(e)
        if header is None:
            header = header_
        else:
            assert header == header_
    print(tabulate(table, headers=header))


if __name__ == "__main__":
    logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    args = Options()
    fill_from_args_(args)
    bm25dpr = BM25forDPR(args.jar, args.anserini_index, args)
    bm25dpr.create(args.train_file, args.output_dir)
