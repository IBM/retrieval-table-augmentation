from util.args_help import fill_from_args
from util.line_corpus import jsonl_lines
import ujson as json
import numpy as np
from typing import List, Callable, Optional
from table_augmentation.augmentation_tasks import Passage
from table_augmentation.normalization import NormalizationStyle


class Options:
    def __init__(self):
        self.predictions = ''
        self.retrieval_gt = ''  # if set we will evaluate retrieval performance at the table level
        self.normalize = NormalizationStyle.identity


class SumMetrics:
    def __init__(self):
        self.sum_rr = 0
        self.total = 0
        self.hit_at = np.zeros(10)

    def add_evaluation(self, correct_rank: int):
        rr = 1.0/correct_rank if correct_rank > 0 else 0
        self.sum_rr += rr
        self.total += 1
        if correct_rank > 0:
            for ndx in range(len(self.hit_at)):
                if ndx+1 >= correct_rank:
                    self.hit_at[ndx] += 1

    def display(self):
        print(f'MRR = {self.sum_rr / self.total}')
        print(f'Hit@K = {self.hit_at / self.total}')


def evaluate(predictions, answer_normalize: Callable[[str], str], retrieval_gt: Optional[str] = None):
    """
    expects jsonl with 'answers' and 'passages'.
    :param predictions:
    :param answer_normalize:
    :param retrieval_gt: optional file with answer bearing ground truth per table
    :return:
    """
    if retrieval_gt:
        table_id2qid_gt = dict()
        for line in jsonl_lines(retrieval_gt):
            jobj = json.loads(line)
            table_id2qid_gt[jobj['table_id']] = jobj['answer_bearing_for']
        table_metrics = SumMetrics()
    else:
        table_id2qid_gt = None
        table_metrics = None

    passage_metrics = SumMetrics()
    for line in jsonl_lines(predictions):
        jobj = json.loads(line)
        answers = [answer_normalize(a) for a in jobj['answers']]
        passages = [Passage.from_dict(p) for p in jobj['passages']]

        if table_id2qid_gt is not None:
            unique_table_ids = []
            for p in passages:
                table_id = p.get_table_id()
                if table_id not in unique_table_ids:
                    unique_table_ids.append(table_id)
            qid = jobj['qid'] if 'qid' in jobj else jobj['id']
            table_rank = -1
            for ndx, tid in enumerate(unique_table_ids):
                if tid in table_id2qid_gt and qid in table_id2qid_gt[tid]:
                    table_rank = ndx + 1
                    break
            table_metrics.add_evaluation(table_rank)
        else:
            table_rank = None

        rank = -1
        for ndx, passage in enumerate(passages):
            if passage.is_answer_bearing(answers, answer_normalize):
                rank = ndx + 1
                break
        if table_rank is not None and rank != -1 and (table_rank == -1 or table_rank > rank):
            print(f'WARNING: {table_rank} > {rank}')
        passage_metrics.add_evaluation(rank)

    print('Passage level answer bearing metrics:')
    passage_metrics.display()
    if table_metrics is not None:
        print('Table level answer bearing metrics:')
        table_metrics.display()


if __name__ == "__main__":
    opts = Options()
    fill_from_args(opts)
    evaluate(opts.predictions, opts.normalize.get_normalizer(), opts.retrieval_gt)
