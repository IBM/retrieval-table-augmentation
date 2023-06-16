from extractive.modeling_candidate_raex import RaexModel, RaexHypers
from util.line_corpus import read_lines
from torch_util.transformer_optimize import TransformerOptimize, LossHistory
import ujson as json
import logging
import torch
from dpr.retriever_dpr import RetrieverDPR
from transformers import AutoTokenizer
from typing import List, Tuple, Union, Optional, Mapping, Callable, Dict
from table_augmentation.table import Table
from table_augmentation.augmentation_tasks import Query
from extractive.extractive_candidates import CandidateTokenizer
from util.reporting import Reporting
import os

logger = logging.getLogger(__name__)


class RaexSystem:
    def __init__(self, hypers: RaexHypers):
        self.hypers = hypers
        if hypers.n_gpu < 1:
            raise ValueError('Must have GPU')
        # reader model
        model = RaexModel(hypers)
        if self.hypers.resume_from:
            model_state_dict = torch.load(os.path.join(self.hypers.resume_from, 'reader', "model.bin"), map_location='cpu')
            model.load_state_dict(model_state_dict)
        model = model.to(hypers.device)
        if self.hypers.train_instances > 0:
            model.train()
            self.optimizer = TransformerOptimize(hypers, hypers.num_train_epochs * hypers.train_instances, model)
            self.loss_history = LossHistory(hypers.train_instances //
                                            (hypers.full_train_batch_size // hypers.gradient_accumulation_steps))
        else:
            self.model = model
            self.model.eval()
        # retriever model
        self.retriever = RetrieverDPR(hypers.dpr, apply_mode=self.hypers.train_instances == 0)  # TODO: later we'll support reranking too
        # tokenize with candidates
        tokenizer = AutoTokenizer.from_pretrained(hypers.model_name_or_path)
        self.candidate_tokenizer = CandidateTokenizer(tokenizer, self.hypers.task.answer_normalization.get_normalizer())
        self.query_maker = self.hypers.task.get_query_maker()
        self.skip_count = 0

    def make_instances(self, jobj: dict, custom_query_maker=None) -> List[Tuple[Union[str, Tuple[str,str]], str, List[str]]]:
        """
        From a table, create a row or column population instance
        :param jobj: json table
        :param custom_query_maker: custom query maker. If the value is None, default query_maker() will be used
        :return: query string or tuple, table_id, answer list
        """
        _query_maker = custom_query_maker if custom_query_maker is not None else self.query_maker
        if self.hypers.is_query:
            queries = [Query.from_dict(jobj)]
        else:
            queries = _query_maker(Table.from_dict(jobj))

        if len(queries) == 0:
            if self.hypers.no_query_is_error:
                logger.error(f'Invalid table for query {jobj["table_id"]}')
                raise ValueError
            else:
                self.skip_count += 1
                return []

        query_tuples = []
        for query in queries:
            if self.hypers.query_single_sequence:
                the_query = query.title + '\n\n' + query.text
            else:
                the_query = (query.title, query.text)
            query_tuples.append((the_query, query.table_id, query.answers))
        return query_tuples

    def close(self):
        self.hypers.dpr.cleanup_corpus_server()
        if self.skip_count > 0:
            logger.info(f'Skipped {self.skip_count} tables with no query possible.')


