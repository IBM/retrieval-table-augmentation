from util.args_help import fill_from_args
from util.line_corpus import jsonl_lines, shuffled_writer
import ujson as json
import logging
import random
from util.reporting import Reporting
from collections import defaultdict
from table_augmentation.augmentation_tasks import Table, TaskOptions, is_answer_bearing, answer_candidates, Query
from typing import Callable, List, Union, Dict
from dpr.retriever_bm25 import BM25Hypers, RetrieverBM25
from dpr.retriever_dpr import DPRHypers, RetrieverDPR

logger = logging.getLogger(__name__)


class Options:
    def __init__(self):
        self.bm25 = BM25Hypers()
        self.bm25.n_docs = 50
        self.dpr = DPRHypers()
        self.dpr.n_docs = 50
        self.dpr.per_gpu_eval_batch_size = 32
        self.task = TaskOptions()
        self.max_answer_expand = -1
        self.max_negatives = 5  # number of negatives to gather for each positive
        self.max_positives = 10  # create at most this many positives per instance


class Args(Options):
    def __init__(self):
        super().__init__()
        self.train_file = ''
        self.output_dir = ''
        self.__required_args__ = ['train_file', 'output_dir', 'task.task']  # NOTE: either dpr or bm25 info will need to be provided too


class RetrieveTrainingForDPR:
    def __init__(self, opts: Options):
        self.opts = opts
        # counts
        self.no_negative_skip_count = 0
        self.no_positive_skip_count = 0
        self.written = 0
        # config
        self.answer_normalizer = opts.task.answer_normalization.get_normalizer()
        self.query_maker = opts.task.get_query_maker()
        self.searchers = []
        self.batch_size = 1024
        # construct the appropriate searcher
        if opts.bm25.anserini_index:
            opts.bm25.allow_fewer_results = True
            self.searchers.append(RetrieverBM25(opts.bm25))
        if opts.dpr.corpus_endpoint:
            opts.dpr._basic_post_init()
            self.searchers.append(RetrieverDPR(opts.dpr, apply_mode=True))
            self.batch_size = min(self.batch_size, opts.dpr.per_gpu_eval_batch_size)
        if len(self.searchers) == 0:
            print('Must supply either a --dpr.corpus_endpoint or a --bm25.anserini_index')
            raise NotImplementedError

    def check_empty(self, query: Query, pos: Union[dict, list], negs: list) -> bool:
        if len(negs) == 0:
            if self.no_negative_skip_count == 0:
                logger.warning(f'No negatives for "{query.title}\n\n{query.text}"\n   Answers: {query.answers}')
            self.no_negative_skip_count += 1
            return True
        if len(pos) == 0:
            if self.no_positive_skip_count == 0:
                logger.warning(f'No positive for "{query.title}\n\n{query.text}"\n   Answers: {query.answers}')
            self.no_positive_skip_count += 1
            return True
        return False

    def merge_docs(self, docs_list: List[List[Dict[str, List[str]]]]) -> List[Dict[str, List[str]]]:
        if len(docs_list) == 1:
            return docs_list[0]
        merged = [{'pid': [], 'title': [], 'text': []} for _ in range(len(docs_list[0]))]
        for docs in docs_list:
            for bi, results in enumerate(docs):
                for pid, title, text in zip(results['pid'], results['title'], results['text']):
                    if pid not in merged[bi]['pid']:
                        merged[bi]['pid'].append(pid)
                        merged[bi]['title'].append(title)
                        merged[bi]['text'].append(text)
        return merged

    def retrieve(self, query_batch: List[Query]) -> List[Dict[str, List[str]]]:
        docs_list = []
        for searcher in self.searchers:
            doc_scores, docs = searcher.retrieve_forward([(q.title, q.text) for q in query_batch],
                                                         exclude_by_pid_prefix=[q.table_id for q in query_batch])[:2]
            docs_list.append(docs)
        return self.merge_docs(docs_list)

    def batch_retrieve_pooled(self, query_batch: List[Query], out):
        docs = self.retrieve(query_batch)
        for query, results in zip(query_batch, docs):
            pos = []
            negs = []
            for pid, title, text in zip(results['pid'], results['title'], results['text']):
                answer_bearing = is_answer_bearing(text, query.answers, self.answer_normalizer)
                if answer_bearing:
                    pos.append({'pid': pid, 'title': title, 'text': text})
                else:
                    negs.append({'pid': pid, 'title': title, 'text': text})
            if self.check_empty(query, pos, negs):
                continue
            out.write(json.dumps({'id': query.qid,
                                  'query': {'title': query.title, 'text': query.text},
                                  'answers': query.answers,
                                  'positives': pos[:self.opts.max_positives],
                                  'negatives': negs[:self.opts.max_negatives]}) + '\n')
            self.written += 1

    def batch_retrieve_by_answer(self, query_batch: List[Query], out):
        docs = self.retrieve(query_batch)
        # docs are list of dict [{title: [t_1...t_n_docs_for_provenance], text: [...], pid: [...]}] * batch_size
        for query, results in zip(query_batch, docs):
            ans2pos = defaultdict(list)
            negs = []
            for pid, title, text in zip(results['pid'], results['title'], results['text']):
                passage = {'pid': pid, 'title': title, 'text': text}
                candidates = [self.answer_normalizer(cand) for cand in answer_candidates(text)]
                answers_found = [ans for ans in query.answers if ans in candidates]
                if len(answers_found) == 0:
                    negs.append(passage)
                else:
                    # CONSIDER: only put example for one instance
                    for ans in answers_found:
                        ans2pos[ans].append(passage)
            # also reduce the ans2pos map to contain at most opts.max_expand_instance
            if len(ans2pos) > self.opts.max_answer_expand:
                trimmed = [(ans, pos) for ans, pos in ans2pos.items()]
                trimmed.sort(key=lambda x: len(x[1]), reverse=True)
                ans2pos = {ans: pos for ans, pos in trimmed[:self.opts.max_answer_expand]}

            if self.check_empty(query, ans2pos, negs):
                continue

            negative_passages = negs[:self.opts.max_negatives * len(ans2pos)]
            for ans, pos in ans2pos.items():
                random.shuffle(negative_passages)
                out.write(json.dumps({'id': f'{query.qid}::{ans}',
                                      'query': {'title': query.title, 'text': query.text},
                                      'answers': query.answers,
                                      'positives': pos[:self.opts.max_positives],
                                      'negatives': negative_passages[:self.opts.max_negatives]}) + '\n')
                self.written += 1

    def batch_retrieve(self, query_batch: List[Query], out):
        if self.opts.max_answer_expand > 0:
            self.batch_retrieve_by_answer(query_batch, out)
        else:
            self.batch_retrieve_pooled(query_batch, out)

    def create(self, train_file, output_dir):
        report = Reporting()

        def display_report(in_progress: bool):
            instance_count = report.check_count * self.batch_size
            if in_progress:
                logger.info(f'On instance {instance_count}, '
                            f'{instance_count / report.elapsed_seconds()} instances per second')
            else:
                logger.info(f'Finished {instance_count} instances; wrote {self.written} training triples. '
                            f'{instance_count / report.elapsed_seconds()} instances per second')
            if self.no_negative_skip_count > 0:
                logger.info(f'{self.no_negative_skip_count} skipped for lack of negatives')
            if self.no_positive_skip_count > 0:
                logger.info(f'{self.no_positive_skip_count} skipped for lack of positives')

        with shuffled_writer(output_dir) as out:
            query_batch = []
            for line in jsonl_lines(train_file):
                jobj = json.loads(line)
                table = Table.from_dict(jobj)
                queries = self.query_maker(table)
                query_batch.extend(queries)
                if len(query_batch) >= self.batch_size:
                    self.batch_retrieve(query_batch[:self.batch_size], out)
                    query_batch = list(query_batch[self.batch_size:])
                    if report.is_time():
                        display_report(True)
            if len(query_batch) > 0:
                self.batch_retrieve(query_batch, out)
            display_report(False)


if __name__ == "__main__":
    logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    args = Args()
    fill_from_args(args)
    bm25dpr = RetrieveTrainingForDPR(args)
    bm25dpr.create(args.train_file, args.output_dir)
