from extractive.modeling_candidate_raex import RaexModel, RaexHypers
from extractive.raex_system import RaexSystem
from util.line_corpus import read_lines, write_open
import ujson as json
import logging
import torch
from typing import List, Tuple, Union, Optional, Mapping, Callable
from util.reporting import Reporting
import os
from collections import Counter
from util.metrics import answer_metrics, aggregate_by_answer_string

logger = logging.getLogger(__name__)


class RaexApplyHypers(RaexHypers):
    def __init__(self):
        super().__init__()
        # CONSIDER:  self.retrieval_gt = ''  # to compute answer-bearing gt at table level
        self.train_instances = 0
        self.__required_args__ = ['model_name_or_path', 'resume_from', 'task.task',
                                  'dpr.corpus_endpoint', 'tables', 'output_dir']


class RaexApplier(RaexSystem):
    def __init__(self, hypers: RaexApplyHypers):
        super().__init__(hypers)
        # metrics
        self.sum_metrics = Counter()
        self.instance_count = 0
        # prediction file
        self.predictions_out = None if self.hypers.output_dir is None \
            else write_open(os.path.join(self.hypers.output_dir, f'{self.hypers.global_rank}.jsonl.gz'))

    def one_instance(self, query: Union[str, Tuple[str, str]], table_id: str, answers: Optional[List[str]]):
        # NOTE: only one query per batch
        doc_scores, docs, ifb = self.retriever.retrieve_forward([query], exclude_by_pid_prefix=[table_id])
        # NOTE: we exclude from retrieval the table we constructed the query from
        assert len(docs) == 1
        if list(doc_scores.shape) != [1, self.hypers.dpr.n_docs]:
            print(f'doc_scores.shape = {doc_scores.shape}: {doc_scores}')
            raise ValueError
        input_tensors, cand_starts, cand_ends, cand_correct, cand_strings = \
            self.candidate_tokenizer.tokenize_with_candidates(query, answers, docs,
                                                              self.hypers.dpr.n_docs, self.hypers.max_seq_length)
        input_tensors = {n: t.to(self.hypers.device) for n, t in input_tensors.items()}
        _, candidate_logits = self.model(
            input_tensors['input_ids'], input_tensors['token_type_ids'], input_tensors['attention_mask'],
            doc_scores.squeeze(0),  # remove batch dimension on doc_scores
            cand_starts, cand_ends, None)
        candidate_logits = candidate_logits.detach().cpu()
        num_answers = len(answers)
        metrics = answer_metrics(cand_strings, candidate_logits, cand_correct, cand_starts[0], num_answers)
        for n, v in metrics.items():
            self.sum_metrics[n] += v
        self.instance_count += 1
        # write out the predictions
        passages = [{'pid': pid, 'title': title, 'text': text, 'score': float(score)}
                    for pid, title, text, score in zip(docs[0]['pid'], docs[0]['title'], docs[0]['text'],
                                                       doc_scores.reshape(-1).detach().cpu().numpy().tolist())]
        predictions = [{'answer': cand_string, 'logit': logit}
                       for cand_string, logit in zip(cand_strings, candidate_logits.numpy().tolist())]
        agg_predictions = [{'answer': cand, 'score': score, 'correct': correct} for cand, score, correct in
                           aggregate_by_answer_string(cand_strings, candidate_logits, cand_correct, max_not_sum=False)]
        if self.predictions_out is not None:
            self.predictions_out.write(json.dumps({'table_id': table_id, 'query': query, 'answers': answers,
                                                   'passages': passages, 'predictions': predictions,
                                                   'aggregated_predictions': agg_predictions}) + '\n')

    def apply(self):
        report = Reporting()
        self.model.eval()
        instance_count = 0
        for jobj in self.hypers.jsonl_instances(self.hypers.tables, rand=None):
            for the_query, table_id, answers in self.make_instances(jobj):
                instance_count += 1
                with torch.no_grad():
                    self.one_instance(the_query, table_id, answers)

                if report.is_time():
                    report.display()
                    display_metrics = [f'{metric_name} = {sum / self.instance_count}' for metric_name, sum in
                                       self.sum_metrics.items()]
                    logger.info('\n'.join(display_metrics))

        display_metrics = [f'{metric_name} = {sum / self.instance_count}' for metric_name, sum in
                           self.sum_metrics.items()]
        logger.info('\n'.join(display_metrics))
        logger.info(f'{instance_count / report.elapsed_seconds()} instances per second. '
                    f'{instance_count} instances, {report.elapsed_time_str()}')

    def close(self):
        super().close()
        self.predictions_out.close()


if __name__ == '__main__':
    hypers_args = RaexApplyHypers()
    hypers_args.fill_from_args()
    applier = RaexApplier(hypers_args)
    applier.apply()
    applier.close()
