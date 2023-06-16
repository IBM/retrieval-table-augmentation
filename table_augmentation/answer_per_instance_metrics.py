import ujson as json
from util.line_corpus import jsonl_lines, write_open
from util.args_help import fill_from_args
from util.metrics import reciprocal_rank, ndcg
import numpy as np
import scipy.stats as st
from collections import defaultdict
import torch


class Options:
    def __init__(self):
        self.predictions = ''
        self.per_instance_file = ''


def evaluate(predictions):
    results = []
    metric_names = ['MRR', 'NDCG10']
    pi_out = write_open(opts.per_instance_file) if opts.per_instance_file else None
    for line in jsonl_lines(predictions):
        jobj = json.loads(line)
        qid = jobj['id'] if 'id' in jobj else jobj['table_id']
        gold_answers = jobj['answers']
        if 'aggregated_predictions' not in jobj:
            per_answer_scores = defaultdict(list)
            for ans in jobj['predictions']:
                per_answer_scores[ans['answer']].append(ans['logit'])
            agg_predictions = []
            for ans, scores in per_answer_scores.items():
                score = torch.logsumexp(torch.tensor(scores), dim=0).item()
                agg_predictions.append({'score': score, 'correct': ans in gold_answers})
            agg_predictions.sort(key=lambda x: x['score'], reverse=True)
        else:
            predictions = jobj['aggregated_predictions']
        correct_ranks = [zrank+1 for zrank, p in enumerate(predictions) if p['correct']]
        rr = reciprocal_rank(correct_ranks)
        ndcg10 = ndcg(correct_ranks, len(jobj['answers']), 10)

        pi_record = {'qid': qid, 'rr': rr, 'ndcg10': ndcg10}
        results.append([rr, ndcg10])
        if pi_out is not None:
            pi_out.write(json.dumps(pi_record)+'\n')
    if pi_out is not None:
        pi_out.close()

    results = np.array(results, dtype=np.float32)
    for metric_ndx in range(results.shape[1]):
        mean = np.mean(results[:, metric_ndx])
        minv, maxv = st.t.interval(0.95, results.shape[0]-1, loc=mean, scale=st.sem(results[:, metric_ndx]))
        print(f'{metric_names[metric_ndx]}: {mean * 100:.2f}, {(mean-minv) * 100:.2f}, {(maxv-mean) * 100:.2f}')


if __name__ == "__main__":
    opts = Options()
    fill_from_args(opts)
    evaluate(opts.predictions)