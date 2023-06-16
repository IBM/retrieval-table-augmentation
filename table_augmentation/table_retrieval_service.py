#!/usr/bin/env python
# encoding: utf-8
from flask import Flask, request, jsonify
import base64
import numpy as np
from util.args_help import fill_from_args
import os
import logging
from dpr.dpr_index import DPRIndex
import random
from transformers import DPRQuestionEncoderTokenizerFast, DPRQuestionEncoder
from table_augmentation.augmentation_tasks import Query, Table, TaskOptions, AugmentationTask, NormalizationStyle
from dpr.dpr_util import queries_to_vectors
import torch
from typing import List, Optional, Callable

logger = logging.getLogger(__name__)


class IndexConfig:
    def __init__(self):
        self.corpus_dir = ''
        self.query_encoder = ''
        self.single_sequence = False


class Options():
    def __init__(self):
        self.port = 5001
        self.col = IndexConfig()
        self.row = IndexConfig()
        self.local_only = False  # only accessible on same machine
        self.debug = False
        self.log_info = False
        self.__required_args__ = ['col.corpus_dir', 'col.query_encoder']


class RetrievalStyle:
    def __init__(self, config: IndexConfig, tokenizer: DPRQuestionEncoderTokenizerFast,
                 query_maker: Callable[[Table], List[Query]]):
        self.dpr_index = DPRIndex(config.corpus_dir, dtype=np.float16)
        self.qencoder = DPRQuestionEncoder.from_pretrained(config.query_encoder)
        self.qencoder = self.qencoder.to('cuda:0')
        self.qencoder.eval()
        self.query_maker = query_maker
        self.tokenizer = tokenizer
        self.title_text_sep = '\n\n'  # only used if opts.query_single_sequence
        self.single_sequence = config.single_sequence

    def query_index(self, tables: List[Table], k: int, include_vectors=False):
        queries = [self.query_maker(t)[0] for t in tables]  # TODO: better error if we can't make a query from the table
        table_ids = [q.table_id for q in queries]
        if self.single_sequence:
            query_strs = [q.title + self.title_text_sep + q.text for q in queries]
        else:
            query_strs = [(q.title, q.text) for q in queries]
        # logger.warning(f'Queries: {query_strs}')
        with torch.no_grad():
            query_vectors = queries_to_vectors(self.tokenizer, self.qencoder, query_strs).detach().cpu().numpy()
        retval = self.dpr_index.retrieve_docs(query_vectors, k,
                                              include_vectors=include_vectors,
                                              exclude_by_pid_prefix=table_ids)
        if include_vectors:
            retval['doc_vectors'] = base64.b64encode(retval['doc_vectors']).decode('ascii')
        return retval


def run(opts: Options):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if opts.log_info else logging.WARNING)
    app = Flask(__name__)
    if not opts.log_info:
        log = logging.getLogger('werkzeug')
        log.disabled = True
        app.logger.disabled = True
        app.logger.setLevel(logging.WARNING)

    tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained('facebook/dpr-question_encoder-multiset-base')

    retrievals = {}

    if opts.col.corpus_dir:
        col = TaskOptions()
        col.no_answers = True
        col.task = AugmentationTask.col
        column_retrieval = RetrievalStyle(opts.col, tokenizer, col.get_query_maker())
        retrievals['col'] = column_retrieval
    if opts.row.corpus_dir:
        row = TaskOptions()
        row.no_answers = True
        row.task = AugmentationTask.row
        row.answer_normalization = NormalizationStyle.lowercase
        row_retrieval = RetrievalStyle(opts.row, tokenizer, row.get_query_maker())
        retrievals['row'] = row_retrieval

    @app.route('/config', methods=['GET'])
    def get_config():
        configs = {}
        dtype = 16
        for name, retriever in retrievals.items():
            config = retriever.dpr_index.get_config()
            assert dtype == config['dtype']
            dtype = config['dtype']
            del config['dtype']
            configs[name] = config
        configs['dtype'] = dtype
        return jsonify(configs)

    @app.route('/retrieve', methods=['POST'])
    def retrieve_tables():
        query = request.get_json()
        # input is two parts:
        #  the list of query tables
        #  k (the number of records per document)
        k = query['k']
        include_vectors = 'include_vectors' in query and query['include_vectors']
        query_tables = [Table.from_dict(t) for t in query['tables']]
        for qt in query_tables:
            qt.candidate_cells[:, :] = 1
        retvals = {}
        for name, retriever in retrievals.items():
            # logger.warning(f'query with {len(query_tables)} tables and k = {k}')
            retvals[name] = retriever.query_index(query_tables, k, include_vectors)
            # TODO: reformat the table passages

        return jsonify(retvals)

    app.run(host='127.0.0.1' if opts.local_only else '0.0.0.0', debug=opts.debug, port=opts.port)


if __name__ == '__main__':
    opts = Options()
    fill_from_args(opts)
    run(opts)
