from util.line_corpus import read_lines, write_open, jsonl_lines
from util.args_help import fill_from_args
from table_augmentation.normalization import NormalizationStyle
from util.metrics import reciprocal_rank, ndcg
from collections import Counter
import ujson as json


class Options:
    def __init__(self):
        self.text_out = ''
        self.target_text = ''
        self.rag_out = ''
        self.beam_size = 35
        self.answer_normalization = NormalizationStyle.identity
        self.per_instance_file = ''
        self.list_output = False


def beam_chunked_lines(text_out: str, beam_size: int, list_output: bool):
    pred_lines = []
    for beam_ndx, beam_line in enumerate(read_lines(text_out)):
        pred_lines.append(beam_line.strip())
        if (beam_ndx+1) % beam_size == 0:
            if list_output:
                yield ('; '.join(pred_lines)).split('; ')
            else:
                yield pred_lines
            pred_lines = []


def bart_predictions(opts: Options, normalizer):
    for preds, target in zip(beam_chunked_lines(opts.text_out, opts.beam_size, opts.list_output),
                             read_lines(opts.target_text)):
        if opts.list_output:
            targets = set([normalizer(a) for a in target.strip().split('; ')])
        else:
            targets = [normalizer(target.strip())]
        yield preds, targets


def rag_predictions(rag_jsonl: str, normalizer):
    for line in jsonl_lines(rag_jsonl):
        jobj = json.loads(line)
        predictions = [p.strip() for p in jobj['predictions']]
        targets = set([normalizer(a) for a in jobj['answers']])
        yield ('; '.join(predictions)).split('; '), targets


if __name__ == '__main__':
    opts = Options()
    fill_from_args(opts)
    normalizer = opts.answer_normalization.get_normalizer()
    metrics = Counter()
    count = 0
    if opts.rag_out:
        assert not opts.text_out and not opts.target_text
        prediction_iter = rag_predictions(opts.rag_out, normalizer)
    else:
        prediction_iter = bart_predictions(opts, normalizer)
    with write_open(opts.per_instance_file) as pi_out:
        for preds, targets in prediction_iter:
            count += 1
            correct_ranks = []
            already_found = []
            for zrank, pred in enumerate(preds):
                npred = normalizer(pred)
                if npred in targets and npred not in already_found:
                    correct_ranks.append(1 + zrank)
                    already_found.append(npred)
            rr = reciprocal_rank(correct_ranks)
            metrics['mrr'] += rr
            ndcg10 = ndcg(correct_ranks, len(targets), 10)
            metrics['ndcg_10'] += ndcg10
            ndcg20 = ndcg(correct_ranks, len(targets), 20)
            metrics['ndcg_20'] += ndcg20
            pi_out.write(json.dumps({'rr': rr, 'ndcg10': ndcg10, 'ndcg20': ndcg20})+'\n')
            # , 'target': targets, 'preds': preds, 'correct_ranks': correct_ranks
    for metric_name, sum in metrics.items():
        print(f'{metric_name} = {sum/count} over {count}')
