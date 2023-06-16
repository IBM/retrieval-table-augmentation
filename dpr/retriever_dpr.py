from torch_util.transformer_optimize import TransformerOptimize
from dpr.retriever_base import RetrieverBase, InfoForBackward
from dpr.dpr_util import tokenize_queries, DPROptions
import torch
import logging
import torch.nn.functional as F
import os
from typing import List, Optional, Union, Tuple, Dict

logger = logging.getLogger(__name__)


class DPRHypers(DPROptions):
    def __init__(self):
        super().__init__()
        self.n_docs = 20
        self.debug = False
        self.max_seq_length = 512
        self.warmup_skip_instances = -1


class DPRInfoForBackward(InfoForBackward):
    def __init__(self, retrieved_doc_embeds, question_encoder_last_hidden_state, input_ids, attention_mask, rng_state):
        super().__init__(rng_state)
        self.retrieved_doc_embeds = retrieved_doc_embeds
        self.question_encoder_last_hidden_state = question_encoder_last_hidden_state
        self.input_ids = input_ids
        self.attention_mask = attention_mask


class RetrieverDPR(RetrieverBase):
    def __init__(self, hypers: DPRHypers, *, apply_mode=False):
        """
        :param hypers:
        """
        super().__init__(hypers)
        self.hypers = hypers
        qencoder, self.tokenizer, self.rest_retriever = hypers.load_model_and_retriever(eval_mode=apply_mode)

        if apply_mode:
            self.optimizer = None
            self.model = qencoder
        else:
            self.optimizer = TransformerOptimize(self.hypers,
                                                 self.hypers.num_train_epochs * self.hypers.train_instances,
                                                 qencoder)
            self.model = self.optimizer.model
        self.backward_count = 0

    def cleanup_corpus_server(self):
        self.hypers.cleanup_corpus_server()

    def retrieve_forward(self, queries: Union[List[str], List[Tuple[str, str]]], *,
                         positive_pids: Optional[List[List[str]]] = None,
                         exclude_by_pid_prefix: Optional[List[str]] = None) -> \
            Tuple[torch.FloatTensor, List[Dict[str, List[str]]], DPRInfoForBackward]:
        """
        :param queries: list of queries to retrieve documents for
        :param positive_pids: used for tracking and reporting on retrieval metrics
        :param exclude_by_pid_prefix: exclude passages with pid starting with indicated value
        :return:
                doc_scores, Tensor (batch x n_docs)
                docs, list of dict [{title: [t_1...t_n_docs_for_provenance], text: [...], pid: [...]}] * batch_size
                info-for-backward (when calling retrieve_backward)
        """
        n_docs = self.hypers.n_docs

        if self.optimizer is not None:
            self.optimizer.model.train()
        else:
            self.model.eval()
        # CONSIDER: we may wish to break this up into multiple calls and multiprocess the REST calls and the GPU encoding
        # NOTE: for KD, we don't actually care about redoing the rng exactly, since the gradient does not depend on it anyway
        rng_state = InfoForBackward.get_rng_state()
        with torch.no_grad():
            # version that supports query title/text
            input_dict = tokenize_queries(self.tokenizer, queries, max_length=self.hypers.max_seq_length)
            #input_dict = prepare_seq2seq_batch(self.tokenizer, queries, return_tensors="pt",
            #                                   max_length=self.hypers.max_seq_length)
            input_ids = input_dict['input_ids'].to(self.model.device)
            attention_mask = input_dict['attention_mask'].to(self.model.device)

            question_encoder_last_hidden_state = self.model(
                input_ids, attention_mask=attention_mask, return_dict=True
            )[0]

            doc_scores, docs, retrieved_doc_embeds = self.rest_retriever.retrieve(
                question_encoder_last_hidden_state, n_docs=n_docs,
                exclude_by_pid_prefix=exclude_by_pid_prefix)
            # NOTE that:
            """
            doc_scores = torch.bmm(
                question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
            ).squeeze(1)
            """

        self.track_retrieval_metrics(positive_pids, docs, display_prefix='DPR:')

        # need to return the retrieved_doc_embeds and the rng state
        # also return the question_encoder_last_hidden_state so we can verify we get the same thing in retrieve_backward
        ifb = DPRInfoForBackward(retrieved_doc_embeds, question_encoder_last_hidden_state,
                                 input_ids, attention_mask, rng_state)

        return doc_scores, docs, ifb

    def add_vectors(self, ifb: DPRInfoForBackward, pids: List[List[str]]):
        """
        In case the reranker is scoring passages from other retrievers, we can also get a KD loss on those
        :param ifb:  returned from retrieve_forward, will be passed to retrieve_backward
        :param pids: the passage ids from the other retrieval methods
        :return: no return, ifb is updated
        """
        _, doc_vectors = self.rest_retriever.fetch(pids)
        doc_vectors = doc_vectors.to(ifb.retrieved_doc_embeds)
        assert len(ifb.retrieved_doc_embeds.shape) == len(doc_vectors.shape) == 3
        assert ifb.retrieved_doc_embeds.shape[0] == doc_vectors.shape[0]
        ifb.retrieved_doc_embeds = torch.cat((ifb.retrieved_doc_embeds, doc_vectors), dim=1)

    def retrieve_backward(self, ifb: DPRInfoForBackward, *,
                          doc_scores_grad: Optional[torch.Tensor] = None,
                          reranker_logits: Optional[torch.Tensor] = None,
                          target_mask: Optional[List[List[bool]]] = None):
        """
        At least one of doc_scores_grad, reranker_logits or target_mask should be provided
        :param ifb: the info-for-backward returned by retrieve_forward
        :param doc_scores_grad: Basic GCP gradients for the doc_scores returned by retrieve_forward
        :param reranker_logits: For KD training the query encoder
        :param target_mask: Ground truth for correct provenance
        :return:
        """
        self.backward_count += ifb.input_ids.shape[0]
        if self.backward_count <= self.hypers.warmup_skip_instances:
            if self.backward_count == self.hypers.warmup_skip_instances:
                logger.info(f'Last skipped DPR backward pass ({self.backward_count})')
            return
        if doc_scores_grad is not None:
            self.optimizer.reporting.report_interval_secs = 10000000  # this loss is artificial, so don't report it
        self.optimizer.model.train()
        # save current rng state and restore forward rng state
        with torch.random.fork_rng(devices=[self.optimizer.hypers.device]):
            ifb.restore_rng_state()
            question_encoder_last_hidden_state = self.optimizer.model(
                ifb.input_ids, attention_mask=ifb.attention_mask, return_dict=True
            )[0]
        # check that question_encoder_last_hidden_state_forward == question_encoder_last_hidden_state
        if self.debug:
            difference = ((question_encoder_last_hidden_state - ifb.question_encoder_last_hidden_state) ** 2).sum()
            if difference > 0.001:
                logger.error(
                    f'the question encoder last hidden state is very different from that computed earlier: {difference}')
                raise ValueError

        retrieved_doc_embeds = ifb.retrieved_doc_embeds.to(question_encoder_last_hidden_state)
        # CONSIDER: support batch negatives here
        doc_scores = torch.bmm(
            question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
        ).squeeze(1)
        if doc_scores_grad is not None:
            grad_loss = torch.dot(doc_scores.reshape(-1), doc_scores_grad.reshape(-1))
            grad_norm_val = torch.linalg.norm(doc_scores_grad.reshape(-1)).item()
        else:
            grad_loss = 0
            grad_norm_val = 0
        if target_mask is not None:
            assert all(type(m) == bool for mb in target_mask for m in mb)
            # TODO: support logsumexp rather than sum for target_mask
            # TODO: target_mask should be a 0/1 FloatTensor, then we can just multiply elementwise, then logsumexp per-instance, then sum
            hard_loss = -(F.log_softmax(doc_scores, dim=1).reshape(-1)[[m for mb in target_mask for m in mb]].sum())
        else:
            hard_loss = 0
        if reranker_logits is not None and self.hypers.kd_alpha > 0.0:
            combined_loss, hard_loss, kd_loss = self.add_kd_loss(doc_scores, reranker_logits, hard_loss + grad_loss)
        else:
            combined_loss = hard_loss + grad_loss
            kd_loss = 0
        self.optimizer.step_loss(combined_loss, grad_norm=grad_norm_val, hard_loss=hard_loss, kd_loss=kd_loss)

    def save(self):
        if self.hypers.global_rank != 0:
            return
        model_to_save = (
            self.optimizer.model.module if hasattr(self.optimizer.model, "module") else self.optimizer.model)
        model_to_save.save_pretrained(os.path.join(self.hypers.output_dir, 'qry_encoder'))
