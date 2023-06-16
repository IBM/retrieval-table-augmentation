import logging
import torch
from torch import nn
from transformers import AutoModel, AutoConfig
from torch_util.hypers_base import HypersBase
from corpus.corpus_client import CorpusClient
from util.line_corpus import read_lines
from dpr.retriever_dpr import DPRHypers
import os
from table_augmentation.augmentation_tasks import TaskOptions

logger = logging.getLogger(__name__)


class RaexHypers(HypersBase):
    def __init__(self):
        super().__init__()
        self.tables = ''
        self.is_query = False  # set flag true if the 'tables' we are reading already are queries
        self.task = TaskOptions()
        self.query_single_sequence = False
        self.train_instances = -1  # -1 to compute from lines in tables files
        self.max_query_length = 128
        self.max_seq_length = 256
        self.retrieve_batch_factor = 1
        self.globally_normalized = False
        self.dpr = DPRHypers()
        self.no_query_is_error = False
        self.__required_args__ = ['model_name_or_path', 'dpr.corpus_endpoint', 'output_dir', 'tables']

    def _post_init(self):
        if not self.dpr.qry_encoder_path and not self.resume_from:
            raise ValueError(f'Must supply dpr query encoder with --dpr.qry_encoder_path or --resume_from')
        self._quiet_post_init = True
        super()._post_init()
        if self.train_instances == -1:
            self.train_instances = sum(1 for _ in read_lines(self.tables))
            logger.info(f'Counted num_instances = {self.train_instances}')
        self.per_gpu_train_batch_size = 1
        self.per_gpu_eval_batch_size = 1
        if self.full_train_batch_size < self.world_size or self.full_train_batch_size % self.world_size != 0:
            raise ValueError(f'full_train_batch_size ({self.full_train_batch_size} '
                             f'must be a multiple of world_size ({self.world_size})')
        self.gradient_accumulation_steps = self.full_train_batch_size // self.world_size
        assert self.n_gpu == 1
        self.dpr.copy_from_base_hypers(self, self.train_instances, per_gpu_batch_size_scale=self.retrieve_batch_factor)
        if self.resume_from and 'dpr.qry_encoder_path' not in self.__passed_args__:
            self.dpr.qry_encoder_path = os.path.join(self.resume_from, 'qry_encoder')

    def cleanup_corpus_server(self):
        CorpusClient.cleanup_corpus_server(self.dpr)


class RaexModel(torch.nn.Module):
    """Model for Retrieval Augmented Extraction with candidates.
    This module is composed of the BERT (or other transformer) model with a linear layer on top of
    the sequence output that computes a probability for each candidate based on its start and end token vectors
    These are gathered across all passages for a query and weighted by the doc_scores for those passages.

    We support multiple correct spans by logsumexping the logits for all valid start/end tokens
    """
    def __init__(self, hypers: RaexHypers):
        super().__init__()
        self.config = AutoConfig.from_pretrained(hypers.model_name_or_path)
        self.transformer = AutoModel.from_pretrained(hypers.model_name_or_path)
        self.start_end_classifier = nn.Linear(self.config.hidden_size*2, 1)
        self.hypers = hypers

    def forward(self, input_ids, token_type_ids, attention_mask, doc_scores,
                cand_starts, cand_ends, cand_correct=None):
        """
        This 'batch' is always the query/context pairs for a SINGLE query. Use gradient accumulation to set batch size.
        :param input_ids:
        :param token_type_ids:
        :param attention_mask:
        :param doc_scores: vector of len n_docs. During training this will have requires_grad
        :param cand_starts: tuple of tensors to index into input_ids where the candidate start special tokens are
        :param cand_ends:   tuple of tensors to index into input_ids where the candidate end special tokens are
        :param cand_correct: list of boolean indicates which candidates are correct
        :return: loss if cand_correct is provided and logits for the candidates
        """
        cand_doc_ndxs = cand_starts[0]  # for each candidate, the index of the document it is in
        num_candidates = len(cand_doc_ndxs)

        # TODO: is this the right way to get sequence_output for any model we could get from AutoModel?
        sequence_output = self.transformer(input_ids=input_ids,
                                           token_type_ids=token_type_ids,
                                           attention_mask=attention_mask)[0]

        # predict for each candidate from the cat of its start token vector and end token vector
        start_vectors = sequence_output[cand_starts]
        # assert list(start_vectors.shape) == [num_candidates, self.config.hidden_size]
        end_vectors = sequence_output[cand_ends]
        candidate_vectors = torch.cat((start_vectors, end_vectors), dim=-1)
        # assert list(candidate_vectors.shape) == [num_candidates, self.config.hidden_size*2]
        cand_logits = self.start_end_classifier(candidate_vectors)
        cand_logits = cand_logits.squeeze(-1)  # == .reshape(-1) since batch_size is 1
        # assert list(cand_logits.shape) == [num_candidates]

        # assert list(doc_scores.shape) == [self.hypers.dpr.n_docs]  # when we use re-ranker it is reranker.n_docs
        doc_scores_normed = torch.log_softmax(doc_scores, dim=0)

        # now weight by doc score
        if self.hypers.globally_normalized:  # NOTE: Not effective for training retrieval but works well for extraction
            expanded_doc_scores = doc_scores_normed[cand_starts[0]]  # duplicate the doc score for each candidate in the doc
            assert expanded_doc_scores.shape == cand_logits.shape
            # combine with doc_scores, then log_softmax over all candidates
            logits_w_doc_score = cand_logits + expanded_doc_scores
            final_logits = torch.log_softmax(logits_w_doc_score, dim=-1)
        else:
            # log_softmax candidates per-document, then combine with doc_scores
            prev_end = 0  # also cur_start
            cur_end = 1
            final_logits = torch.zeros_like(cand_logits)
            while cur_end <= len(cand_logits):
                # if we are at the end or switching to a new document
                if cur_end == len(cand_logits) or cand_doc_ndxs[cur_end] != cand_doc_ndxs[prev_end]:
                    # we will logsoftmax over the candidates in the current document, and add the doc_score
                    final_logits[prev_end:cur_end] = torch.log_softmax(cand_logits[prev_end:cur_end], dim=0) + \
                                                     doc_scores_normed[cand_doc_ndxs[prev_end]]
                    prev_end = cur_end
                cur_end += 1
            # verify valid prob. dist. over candidates
            # assert abs(torch.logsumexp(final_logits, dim=0).item()) < 0.01

        if cand_correct is not None:
            log_like = torch.logsumexp(final_logits[cand_correct], dim=-1)
            loss = -log_like
        else:
            loss = None
        return loss, final_logits
