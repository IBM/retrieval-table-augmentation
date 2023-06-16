from util.line_corpus import read_lines, write_open
import ujson as json
from util.reporting import Reporting, Distribution
import logging
from dpr.retriever_bm25 import BM25Hypers, RetrieverBM25
from util.args_help import fill_from_args
from table_augmentation.eval_answer_bearing import evaluate
from typing import Tuple, List, Dict
from table_augmentation.augmentation_tasks import Query, TaskOptions
from table_augmentation.table import Table

logger = logging.getLogger(__name__)


class Options(BM25Hypers):
    def __init__(self):
        super().__init__()
        self.output = ''
        self.tables = ''
        self.task = TaskOptions()
        self._required_args = ['task.task', 'tables', 'output', 'anserini_index', 'jar']

    def _post_argparse(self):
        self.allow_fewer_results = True  # we aren't passing to batched processing so we can have variable results


opts = Options()
fill_from_args(opts)
retriever = RetrieverBM25(opts)
report = Reporting()
hit_counts = Distribution(thresholds=(0, 1, 3, 5, 10), num_samples_per_threshold=0)


def retrieve(queries: List[Tuple[str, str]], table_ids: List[str]):
    doc_scores, docs = retriever.retrieve_forward(queries, exclude_by_pid_prefix=table_ids)
    retrieved_doc_ids = [dd['pid'] for dd in docs]
    passages = [{'titles': dd['title'], 'texts': dd['text']} for dd in docs]
    assert type(retrieved_doc_ids) == list
    assert all([type(doc_ids) == list for doc_ids in retrieved_doc_ids])
    if not all([type(doc_id) == str for doc_ids in retrieved_doc_ids for doc_id in doc_ids]):
        print(f'Error: {retrieved_doc_ids}')
        raise ValueError('not right type')
    return retrieved_doc_ids, passages


def record_one_instance(output, query: Query, doc_ids: List[str], passages: Dict[str, List[str]]):
    pred_record = {'id': query.qid, 'table_id': query.table_id,
                   'query': {'title': query.title, 'text': query.text},
                   'pids': doc_ids, 'answers': query.answers}
    hit_counts.note_value(len(doc_ids))
    if passages:
        pred_record['passages'] = [{'pid': pid, 'title': title, 'text': text}
                                   for pid, title, text in zip(doc_ids, passages['titles'], passages['texts'])]
    output.write(json.dumps(pred_record) + '\n')
    if report.is_time():
        print(f'Finished instance {report.check_count}; {report.check_count/report.elapsed_seconds()} per second.')


def one_batch(query_batch: List[Query], output):
    """
    retrieve and record one batch of queries
    :param query_batch:
    :param output:
    :return:
    """
    retrieved_doc_ids, passages = retrieve([(q.title, q.text) for q in query_batch], [q.table_id for q in query_batch])
    for bi in range(len(query_batch)):
        record_one_instance(output, query_batch[bi], retrieved_doc_ids[bi], passages[bi] if passages else None)


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
        if len(query_batch) == 2 * opts.num_processes:
            one_batch(query_batch, output)
            query_batch = []
    if len(query_batch) > 0:
        one_batch(query_batch, output)
    print(f'Finished instance {report.check_count}; {report.check_count/report.elapsed_seconds()} per second.')

print(f'Hit counts:')
hit_counts.display(show_sums=False)
print(f'Skipped {skip_count}')
evaluate(opts.output, opts.task.answer_normalization.get_normalizer())
