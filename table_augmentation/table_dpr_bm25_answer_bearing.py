from util.args_help import fill_from_args
from util.line_corpus import jsonl_lines, shuffled_writer
import jnius_config
import ujson as json
import multiprocessing
from multiprocessing.pool import ThreadPool
import functools
import logging
import random
from util.reporting import Reporting
from collections import defaultdict
from table_augmentation.augmentation_tasks import Table, TaskOptions, is_answer_bearing, answer_candidates
from typing import Callable
from dpr.anserini_index import anserini_index, pyserini_index
from pyserini.search.lucene import LuceneSearcher

logger = logging.getLogger(__name__)


class Options:
    def __init__(self):
        self.num_processes = multiprocessing.cpu_count()
        self.task = TaskOptions()
        self.max_answer_expand = -1
        self.num_hits = 50
        self.max_negatives = 5  # number of negatives to gather for each positive
        self.max_positives = 10  # create at most this many training instances per original instance (ignoring the other positives)


class Args(Options):
    def __init__(self):
        super().__init__()
        self.jar = ''
        self.anserini_index = ''
        self.train_file = ''
        self.output_dir = ''
        self.__required_args__ = ['anserini_index', 'train_file', 'output_dir', 'task.task']


def _retrieve_one(query_tuple, searcher, JString, num_hits: int, answer_normalizer: Callable[[str], str]):
    inst_id, query_title, query_text, answers, table_id = query_tuple
    query = query_title + '\n\n' + query_text
    if type(searcher) is LuceneSearcher:
        hits = searcher.search(query, num_hits)
    else:
        hits = searcher.search(JString(query.encode('utf-8')), num_hits)
    pos = []
    negs = []
    for hit in hits:
        pid = hit.docid
        if pid.startswith(table_id):
            continue
        hit_obj = json.loads(hit.raw)
        title_sep_index = hit_obj["contents"].find('\n\n')
        title = hit_obj["contents"][:title_sep_index]
        text = hit_obj["contents"][title_sep_index + 2:]

        answer_bearing = is_answer_bearing(text, answers, answer_normalizer)
        if answer_bearing:
            pos.append({'pid': pid, 'title': title, 'text': text})
        else:
            negs.append({'pid': pid, 'title': title, 'text': text})
    return pos, negs


def _retrieve_by_answer(query_tuple, searcher, JString, num_hits: int,
                        answer_normalizer: Callable[[str], str], max_answer_expand: int):
    inst_id, query_title, query_text, answers, table_id = query_tuple
    query = query_title + '\n\n' + query_text
    if type(searcher) is LuceneSearcher:
        hits = searcher.search(query, num_hits)
    else:
        hits = searcher.search(JString(query.encode('utf-8')), num_hits)
    ans2pos = defaultdict(list)
    negs = []
    for hit in hits:
        pid = hit.docid
        if pid.startswith(table_id):
            continue
        hit_obj = json.loads(hit.raw)
        title_sep_index = hit_obj["contents"].find('\n\n')
        title = hit_obj["contents"][:title_sep_index]
        text = hit_obj["contents"][title_sep_index + 2:]
        passage = {'pid': pid, 'title': title, 'text': text}
        candidates = [answer_normalizer(cand) for cand in answer_candidates(text)]
        answers_found = [ans for ans in answers if ans in candidates]
        if len(answers_found) == 0:
            negs.append(passage)
        else:
            # CONSIDER: only put example for one instance
            for ans in answers_found:
                ans2pos[ans].append(passage)
    # also reduce the ans2pos map to contain at most opts.max_expand_instance
    if len(ans2pos) > max_answer_expand:
        trimmed = [(ans, pos) for ans, pos in ans2pos.items()]
        trimmed.sort(key=lambda x:len(x[1]), reverse=True)
        ans2pos = {ans: pos for ans, pos in trimmed[:max_answer_expand]}
    return ans2pos, negs


class BM25forDPR:
    def __init__(self, opts: Options):
        if opts.jar:
            from jnius import autoclass
            self.JString = autoclass('java.lang.String')
            JSearcher = autoclass('io.anserini.search.SimpleSearcher')
            self.searcher = JSearcher(self.JString(opts.anserini_index))
        else:
            self.JString = None
            self.searcher = LuceneSearcher(opts.anserini_index)

        self.opts = opts
        # NOTE: only thread-based pooling works with the JSearcher
        self.pool = ThreadPool(processes=opts.num_processes)
        logger.info(f'Using multiprocessing pool with {opts.num_processes} workers')
        self.no_negative_skip_count = 0
        self.no_positive_skip_count = 0
        answer_normalizer = opts.task.answer_normalization.get_normalizer()
        self.query_maker = opts.task.get_query_maker()
        if opts.max_answer_expand > 1:
            self._retrieve_one = functools.partial(_retrieve_by_answer,
                                                   searcher=self.searcher, JString=self.JString,
                                                   num_hits=opts.num_hits, answer_normalizer=answer_normalizer,
                                                   max_answer_expand=opts.max_answer_expand)
        else:
            self._retrieve_one = functools.partial(_retrieve_one,
                                                   searcher=self.searcher, JString=self.JString,
                                                   num_hits=opts.num_hits, answer_normalizer=answer_normalizer)
        self.written = 0

    def _write_batch(self, out, query_tuples, passages):
        for query_tuple, passage_pos_neg in zip(query_tuples, passages):
            inst_id, query_title, query_text, answers, table_id = query_tuple
            positive_passages, negative_passages = passage_pos_neg
            if len(negative_passages) == 0:
                if self.no_negative_skip_count == 0:
                    logger.warning(f'No negatives for "{query_title}\n\n{query_text}"\n   Answers: {answers}')
                self.no_negative_skip_count += 1
                continue
            if len(positive_passages) == 0:
                if self.no_positive_skip_count == 0:
                    logger.warning(f'No positive for "{query_title}\n\n{query_text}"\n   Answers: {answers}')
                self.no_positive_skip_count += 1
                continue
            if isinstance(positive_passages, dict):
                # we have a list of positive passages for each answer in recall
                negative_passages = negative_passages[:self.opts.max_negatives*len(positive_passages)]
                for ans, pos in positive_passages:
                    random.shuffle(negative_passages)
                    out.write(json.dumps({'id': f'{inst_id}::{ans}',
                                          'query': {'title': query_title, 'text': query_text},
                                          'answers': answers,
                                          'positives': pos[:self.opts.max_positives],
                                          'negatives': negative_passages[:self.opts.max_negatives]}) + '\n')
            else:
                # we have a single unified list of positive passages
                out.write(json.dumps({'id': inst_id,
                                      'query': {'title': query_title, 'text': query_text},
                                      'answers': answers,
                                      'positives': positive_passages[:self.opts.max_positives],
                                      'negatives': negative_passages[:self.opts.max_negatives]}) + '\n')
            self.written += 1

    def create(self, train_file, output_dir):
        report = Reporting()
        batch_size = 1024
        rand = random.Random()
        with shuffled_writer(output_dir) as out:
            query_tuples = []
            for line in jsonl_lines(train_file):
                jobj = json.loads(line)
                """
                {"pid":"table-0875-912::0",
                "title":"2011 FIBA Asia Champions Cup\tStatistical leaders",
                "text":"Pos.: 1 * Name: Osama Daghles * G: 7 * Asts.: 52 * APG: 7.4"}
                
                {"table_id":"table-0438-806","title":"1935\u201336 Montreal Canadiens season\nRegular season",
                "header":["Player","Pos","GP","G","A","Pts","PIM"],
                "rows":[["Leroy Goldsworthy","RW","47","15","11","26","8"],
                        ["Paul Haynes (ice hockey)","C","48","5","19","24","24"],
                        ["Aur\u00e8le Joliat","LW","48","15","8","23","16"],
                        ["Jack McGill (ice hockey b. 1909)","LW","46","13","7","20","28"],
                        ["Armand Mondou","LW","36","7","11","18","10"],
                        ["Johnny Gagnon","RW","48","7","9","16","42"],
                        ["Alfred L\u00e9pine","C","32","6","10","16","4"]],}
                """
                # make column query or make row query
                table = Table.from_dict(jobj)
                queries = self.query_maker(table)
                if len(queries) == 0:
                    continue
                # TODO: update query_tuples to be query_batch: List[Query], we can just extend with queries
                for query in queries:
                    query_tuples.append((query.qid, query.title, query.text, query.answers, query.table_id))
                if len(query_tuples) >= batch_size:
                    passages = self.pool.map(self._retrieve_one, query_tuples)
                    self._write_batch(out, query_tuples, passages)
                    query_tuples = []
                    if report.is_time():
                        instance_count = report.check_count*batch_size
                        logger.info(f'On instance {instance_count}, '
                                    f'{instance_count/report.elapsed_seconds()} instances per second')
                        if self.no_negative_skip_count > 0:
                            logger.info(f'{self.no_negative_skip_count} skipped for lack of negatives')
                        if self.no_positive_skip_count > 0:
                            logger.info(f'{self.no_positive_skip_count} skipped for lack of positives')

            if len(query_tuples) > 0:
                passages = self.pool.map(self._retrieve_one, query_tuples)
                self._write_batch(out, query_tuples, passages)
            instance_count = report.check_count * batch_size
            logger.info(f'Finished {instance_count} instances; wrote {self.written} training triples. '
                        f'{instance_count/report.elapsed_seconds()} instances per second')
            if self.no_negative_skip_count > 0:
                logger.info(f'{self.no_negative_skip_count} skipped for lack of negatives')
            if self.no_positive_skip_count > 0:
                logger.info(f'{self.no_positive_skip_count} skipped for lack of positives')
            self.pool.close()
            self.pool.join()


if __name__ == "__main__":
    logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    args = Args()
    fill_from_args(args)
    bm25dpr = BM25forDPR(args)
    bm25dpr.create(args.train_file, args.output_dir)

