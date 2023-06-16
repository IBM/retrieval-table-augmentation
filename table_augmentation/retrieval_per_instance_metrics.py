import ujson as json
from util.line_corpus import jsonl_lines, write_open
from util.args_help import fill_from_args
from table_augmentation.normalization import NormalizationStyle
from typing import Callable, List, Optional
from table_augmentation.augmentation_tasks import Passage
import numpy as np
import scipy.stats as st


class Options:
    def __init__(self):
        self.predictions = ''
        self.answer_gt = ''
        self.per_instance_file = ''
        self.answer_normalization = NormalizationStyle.identity
        self.retrieval_gt = ''


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


def evaluate(predictions, answer_normalize: Callable[[str], str], retrieval_gt: Optional[str] = None, answer_gt: Optional[str] = None):
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

    id2answers = None
    if answer_gt:
        id2answers = dict()
        for line in jsonl_lines(answer_gt):
            jobj = json.loads(line)
            qid = jobj['id']
            answers = [answer_normalize(a) for a in jobj['answers']]
            id2answers[qid] = answers

    passage_metrics = SumMetrics()
    results = []
    metric_names = ['MRR']
    with write_open(opts.per_instance_file) as pi_out:
        for line in jsonl_lines(predictions):
            jobj = json.loads(line)
            qid = jobj['id']
            if id2answers is not None:
                assert qid in id2answers
                answers = id2answers[qid]
            else:
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
            rr = 1.0 / rank if rank > 0 else 0.0
            pi_record = {'qid': qid, 'rr': rr}
            results.append([rr])
            if table_rank is not None:
                pi_record['table_rr'] = 1.0 / table_rank if table_rank > 0 else 0.0
            pi_out.write(json.dumps(pi_record)+'\n')

    print('Passage level answer bearing metrics:')
    passage_metrics.display()
    if table_metrics is not None:
        print('Table level answer bearing metrics:')
        table_metrics.display()
    results = np.array(results, dtype=np.float32)
    for metric_ndx in range(results.shape[1]):
        mean = np.mean(results[:, metric_ndx])
        minv, maxv = st.t.interval(0.95, results.shape[0]-1, loc=mean, scale=st.sem(results[:, metric_ndx]))
        print(f'{metric_names[metric_ndx]}: {mean * 100:.2f}, {(mean-minv) * 100:.2f}, {(maxv-mean) * 100:.2f}')


if __name__ == "__main__":
    opts = Options()
    fill_from_args(opts)
    evaluate(opts.predictions, opts.answer_normalization.get_normalizer(), opts.retrieval_gt, opts.answer_gt)