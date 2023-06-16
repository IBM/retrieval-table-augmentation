from util.line_corpus import read_lines
from util.args_help import fill_from_args
from table_augmentation.normalization import NormalizationStyle
from util.metrics import reciprocal_rank, ndcg
from collections import Counter


class Options:
    def __init__(self):
        self.text_out = ''
        self.target_text = ''
        self.beam_size = 35
        self.answer_normalization = NormalizationStyle.identity


def beam_chunked_lines(text_out: str, beam_size: int):
    pred_lines = []
    for beam_ndx, beam_line in enumerate(read_lines(text_out)):
        pred_lines.append(beam_line.strip())
        if (beam_ndx+1) % beam_size == 0:
            yield '; '.join(pred_lines)
            pred_lines = []


if __name__ == '__main__':
    opts = Options()
    fill_from_args(opts)
    normalizer = opts.answer_normalization.get_normalizer()
    metrics = Counter()
    count = 0
    for preds, target in zip(beam_chunked_lines(opts.text_out, opts.beam_size), read_lines(opts.target_text)):
        preds = preds.split('; ')
        targets = set([normalizer(a) for a in target.strip().split('; ')])
        count += 1
        correct_ranks = []
        already_found = []
        for zrank, pred in enumerate(preds):
            npred = normalizer(pred)
            if npred in targets and npred not in already_found:
                correct_ranks.append(1 + zrank)
                already_found.append(npred)
        metrics['mrr'] += reciprocal_rank(correct_ranks)
        metrics['ndcg_10'] += ndcg(correct_ranks, len(targets), 10)
        metrics['ndcg_20'] += ndcg(correct_ranks, len(targets), 20)
    for metric_name, sum in metrics.items():
        print(f'{metric_name} = {sum/count} over {count}')
