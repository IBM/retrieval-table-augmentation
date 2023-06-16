import torch
from util.line_corpus import read_lines, write_open
import ujson as json
from util.reporting import Reporting
import logging
from dpr.dpr_util import DPROptions, queries_to_vectors
from table_augmentation.eval_answer_bearing import evaluate
from table_augmentation.augmentation_tasks import TaskOptions, Query
from table_augmentation.table import Table
import os
from typing import List, Tuple, Optional, Dict, Union

logger = logging.getLogger(__name__)


class Options(DPROptions):
    def __init__(self):
        super().__init__()
        self.output = ''
        self.tables = ''
        self.corpus_endpoint = ''
        self.qry_encoder_path = ''
        self.rag_model_path = ''
        self.task = TaskOptions()
        self.query_single_sequence = False
        self.n_docs_for_provenance = 20  # we'll supply this many document ids for reporting provenance
        self.retrieve_batch_size = 32
        self.__required_args__ = ['task.task', 'tables', 'output', 'corpus_endpoint']


opts = Options()
opts.fill_from_args()
torch.set_grad_enabled(False)

qencoder, tokenizer, rest_retriever = opts.load_model_and_retriever()
report = Reporting()
title_text_sep = '\n\n'  # only used if opts.query_single_sequence


def retrieve(queries: Union[List[str], List[Tuple[str, str]]], table_ids: List[str]):
    with torch.no_grad():
        query_vectors = queries_to_vectors(tokenizer, qencoder, queries)
        doc_scores, docs, doc_vectors = rest_retriever.retrieve(query_vectors, n_docs=opts.n_docs_for_provenance,
                                                                exclude_by_pid_prefix=table_ids)
        doc_scores = doc_scores.detach().cpu().numpy()

    retrieved_doc_ids = [dd['pid'] for dd in docs]
    passages = [{'titles': dd['title'], 'texts': dd['text'], 'scores': doc_scores[dndx].tolist()}
                for dndx, dd in enumerate(docs)]
    return retrieved_doc_ids, passages


def record_one_instance(output, query: Query, doc_ids: List[str], passages: Dict[str, List[str]]):
    pred_record = {'id': query.qid, 'table_id': query.table_id,
                   'query': {'title': query.title, 'text': query.text},
                   'pids': doc_ids, 'answers': query.answers,
                   'passages': [{'pid': pid, 'title': title, 'text': text, 'score': float(score)}
                                for pid, title, text, score in zip(doc_ids, passages['titles'],
                                                                   passages['texts'], passages['scores'])]}
    output.write(json.dumps(pred_record) + '\n')
    if report.is_time():
        print(f'Finished instance {report.check_count}; {report.check_count/report.elapsed_seconds()} per second. '
              f'{1000 * rest_retriever.retrieval_time/report.check_count} msecs per retrieval.')


def one_batch(query_batch: List[Query], output):
    if opts.query_single_sequence:
        queries = [q.title + title_text_sep + q.text for q in query_batch]
    else:
        queries = [(q.title, q.text) for q in query_batch]
    retrieved_doc_ids, passages = retrieve(queries, [q.table_id for q in query_batch])
    for bi in range(len(query_batch)):
        record_one_instance(output, query_batch[bi], retrieved_doc_ids[bi], passages[bi])


if opts.world_size > 1:
    raise ValueError('Distributed not supported')
skip_count = 0
query_maker = opts.task.get_query_maker()
with write_open(opts.output) as output:
    query_batch = []
    for line_ndx, line in enumerate(read_lines(opts.tables)):
        jobj = json.loads(line)
        table = Table.from_dict(jobj)
        queries = query_maker(table)
        if len(queries) == 0:
            if skip_count == 0:
                logger.warning(f'No query possible for {jobj}')
            skip_count += 1
            continue
        query_batch.extend(queries)
        if len(query_batch) >= opts.retrieve_batch_size:
            one_batch(query_batch[:opts.retrieve_batch_size], output)
            query_batch = list(query_batch[opts.retrieve_batch_size:])
    if len(query_batch) > 0:
        one_batch(query_batch, output)
    print(f'Finished instance {report.check_count}; {report.check_count/report.elapsed_seconds()} per second. '
          f'{1000 * rest_retriever.retrieval_time/report.check_count} msecs per retrieval.')

opts.cleanup_corpus_server()
print(f'Skipped {skip_count}')
evaluate(opts.output, opts.task.answer_normalization.get_normalizer())