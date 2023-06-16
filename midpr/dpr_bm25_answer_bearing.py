from util.args_help import fill_from_args
from util.line_corpus import jsonl_lines, shuffled_writer
import jnius_config
import ujson as json
import re
import multiprocessing
from multiprocessing.pool import ThreadPool
import functools
import logging
from util.reporting import Reporting

logger = logging.getLogger(__name__)


class Options:
    def __init__(self):
        self.num_processes = multiprocessing.cpu_count()
        self.max_negatives = 5  # number of negatives to gather for each positive
        self.max_positives = 10  # create at most this many training instances per original instance (ignoring the other positives)
        self.candidates_marked = False  # if true, answer candidate spans are marked with '«' and '»'


class Args(Options):
    def __init__(self):
        super().__init__()
        self.jar = ''
        self.anserini_index = ''
        self.train_file = ''
        self.output_dir = ''
        self.__required_args__ = ['anserini_index', 'train_file', 'jar', 'output_dir']


"""
_NON_WORD = re.compile(r'\W+')


def normalize(text: str):
    normed = _NON_WORD.sub(' ', text).lower().strip()
    if len(normed) > 0:
        return normed
    else:
        return text.strip()
"""


def normalize(text: str):
    return text.lower().strip()


def _retrieve_one(query_tuple, searcher, JString):
    inst_id, query, answers = query_tuple
    hits = searcher.search(JString(query.encode('utf-8')), 50)
    pos = []
    negs = []
    for hit in hits:
        pid = hit.docid
        title = hit.content[:hit.content.find('\n\n')]
        text = hit.content[hit.content.find('\n\n') + 2:]
        norm_context = ' ' + normalize(title + ' ' + text) + ' '
        answer_bearing = any([ans in norm_context for ans in answers])
        # TODO: group the passages by answer; we can return a dict of positive and negative bags
        #   each positive bag is for a single answer
        if answer_bearing:
            pos.append({'pid': pid, 'title': title, 'text': text})
        else:
            negs.append({'pid': pid, 'title': title, 'text': text})
    return pos, negs


class BM25forDPR:
    def __init__(self, jar: str, index: str, opts: Options):
        # TODO: we should use retriever_base, so we can do BM25 negatives or DPR negatives
        jnius_config.set_classpath(jar)
        from jnius import autoclass
        self.JString = autoclass('java.lang.String')
        JSearcher = autoclass('io.anserini.search.SimpleSearcher')
        self.searcher = JSearcher(self.JString(index))
        self.opts = opts
        # NOTE: only thread-based pooling works with the JSearcher
        self.pool = ThreadPool(processes=opts.num_processes)
        logger.info(f'Using multiprocessing pool with {opts.num_processes} workers')
        self.no_negative_skip_count = 0
        self.no_positive_skip_count = 0
        self._retrieve_one = functools.partial(_retrieve_one,
                                               searcher=self.searcher, JString=self.JString)
        self.written = 0

    def _write_batch(self, out, query_tuples, passages):
        for query_tuple, passage_pos_neg in zip(query_tuples, passages):
            inst_id, query, answers = query_tuple
            positive_passages, negative_passages = passage_pos_neg
            if len(negative_passages) == 0:
                if self.no_negative_skip_count == 0:
                    logger.warning(f'No negatives for "{query}"\n   Answers: {answers}')
                self.no_negative_skip_count += 1
                continue
            if len(positive_passages) == 0:
                if self.no_positive_skip_count == 0:
                    logger.warning(f'No positive for "{query}"\n   Answers: {answers}')
                self.no_positive_skip_count += 1
                continue
            # CONSIDER: will also need to take care with conflict-free-batches
            out.write(json.dumps({'id': inst_id, 'query': query,
                                  'answers': answers,
                                  'positives': positive_passages[:self.opts.max_positives],
                                  'negatives': negative_passages[:self.opts.max_negatives]}) + '\n')
            self.written += 1

    def create(self, train_file, output_dir):
        report = Reporting()
        batch_size = 1024
        with shuffled_writer(output_dir) as out:
            query_tuples = []
            for line in jsonl_lines(train_file):
                jobj = json.loads(line)
                inst_id = jobj['id']
                query = jobj['query']
                answers = list(set([normalize(a) for a in jobj['answers']]))
                if self.opts.candidates_marked:
                    answers = ['«'+a+'»' for a in answers]
                query_tuples.append((inst_id, query, answers))
                if len(query_tuples) >= batch_size:
                    passages = self.pool.map(self._retrieve_one, query_tuples)
                    # TODO: maybe pass the original, unnormalized answers
                    self._write_batch(out, query_tuples, passages)
                    query_tuples = []
                    if report.is_time():
                        instance_count = report.check_count*batch_size
                        logger.info(f'On instance {instance_count}, '
                                    f'{instance_count/report.elapsed_seconds()} instances per second')
                        if self.no_negative_skip_count > 0:
                            logger.info(f'{self.no_negative_skip_count} skipped for lack of negatives')

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


if __name__ == "__main__":
    logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    args = Args()
    fill_from_args(args)
    bm25dpr = BM25forDPR(args.jar, args.anserini_index, args)
    bm25dpr.create(args.train_file, args.output_dir)
