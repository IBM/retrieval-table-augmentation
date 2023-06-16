from extractive.modeling_candidate_raex import RaexModel, RaexHypers
from extractive.raex_system import RaexSystem
import random
import logging
import torch
from typing import List, Tuple, Union, Optional
from util.reporting import Reporting
import os
from util.metrics import answer_metrics

logger = logging.getLogger(__name__)


class RaexTrainer(RaexSystem):
    def __init__(self, hypers: RaexHypers):
        super().__init__(hypers)
        self.instance_seen_count = 0
        self.instance_skip_count = 0

    def one_instance(self, query: Union[str, Tuple[str, str]], table_id: str, answers: List[str]) -> bool:
        # NOTE: only one query per batch
        doc_scores, docs, ifb = self.retriever.retrieve_forward([query], exclude_by_pid_prefix=[table_id])
        # NOTE: we exclude from retrieval the table we constructed the query from
        assert len(docs) == 1
        assert list(doc_scores.shape) == [1, self.hypers.dpr.n_docs]
        input_tensors, cand_starts, cand_ends, cand_correct, cand_strings = \
            self.candidate_tokenizer.tokenize_with_candidates(query, answers, docs,
                                                              self.hypers.dpr.n_docs, self.hypers.max_seq_length)
        input_tensors = {n: t.to(self.hypers.device) for n, t in input_tensors.items()}
        num_answers = len(answers)
        if sum(cand_correct) == 0:
            self.optimizer.reporting.moving_averages(**answer_metrics(cand_strings, None, cand_correct,
                                                                      cand_starts[0], num_answers))
            return False  # we skip instances where the answer is not in recall
        doc_scores.requires_grad = True
        loss, candidate_logits = self.optimizer.model(
                    input_tensors['input_ids'], input_tensors['token_type_ids'], input_tensors['attention_mask'],
                    doc_scores.squeeze(0),  # remove batch dimension on doc_scores
                    cand_starts, cand_ends, cand_correct)

        self.loss_history.note_loss(loss.item())
        metrics = answer_metrics(cand_strings, candidate_logits.detach().cpu(), cand_correct, cand_starts[0], num_answers)
        metrics['skip_fraction'] = self.instance_skip_count/self.instance_seen_count
        self.optimizer.step_loss(loss, **metrics)
        self.retriever.retrieve_backward(ifb, doc_scores_grad=doc_scores.grad.detach() * self.hypers.dpr.gradient_accumulation_steps)
        return True

    def train(self):
        rand = random.Random(self.hypers.seed)
        report = Reporting()
        on_epoch = 0
        while True:
            self.optimizer.model.train()
            on_epoch += 1
            for jobj in self.hypers.jsonl_instances(self.hypers.tables, rand=rand):
                for the_query, table_id, answers in self.make_instances(jobj):
                    self.instance_seen_count += 1
                    if not self.one_instance(the_query, table_id, answers):
                        self.instance_skip_count += 1
                    if report.is_time():
                        logger.info(f'Skipped {self.instance_skip_count} out of {self.instance_seen_count} '
                                    f'due to no answer in recall')
                    if not self.optimizer.should_continue():
                        return
            self.save()  # save per-epoch

    def save(self):
        if self.hypers.world_size > 1:
            torch.distributed.barrier()
        if self.hypers.global_rank == 0:
            os.makedirs(os.path.join(self.hypers.output_dir, 'reader'), exist_ok=True)
            logger.info(f'Saving model to {self.hypers.output_dir}')
            model_to_save = (
                self.optimizer.model.module if hasattr(self.optimizer.model, 'module') else self.optimizer.model
            )  # Take care of distributed/parallel training
            torch.save(self.hypers, os.path.join(self.hypers.output_dir, 'reader', 'training_args.bin'))
            torch.save(model_to_save.state_dict(), os.path.join(self.hypers.output_dir, 'reader', 'model.bin'))
        self.retriever.save()
        logger.info(f'loss_history = {self.loss_history.loss_history}')


if __name__ == '__main__':
    hypers_args = RaexHypers()
    hypers_args.fill_from_args()
    hypers_args.set_seed()
    trainer = RaexTrainer(hypers_args)
    trainer.train()
    trainer.save()
    trainer.close()

