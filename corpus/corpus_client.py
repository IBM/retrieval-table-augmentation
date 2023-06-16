import requests
import base64
import numpy as np
import json
import torch
import signal
import logging
import time
import os
from torch_util.hypers_base import HypersBase
from util.reporting import Reporting
from typing import List, Optional
from dpr.dpr_index import DPRIndex

logger = logging.getLogger(__name__)


class CorpusClient:
    def __init__(self, hypers: HypersBase):
        """
        @param hypers: we also assume this has the special attribute 'corpus_endpoint'
        """
        self.dpr_index: Optional[DPRIndex] = None
        self._ensure_server(hypers)
        if self.dpr_index is None:
            self.endpoint = hypers.corpus_endpoint  # 'http://localhost:5001'
            self.headers = {'Content-Type': 'application/json'}
            # get config info from server
            config = requests.get(self.endpoint+'/config', headers=self.headers).json()
            self.rest_dtype = np.float32 if config['dtype'] == 32 else np.float16
        else:
            self.endpoint = None
            self.headers = None
            self.rest_dtype = None
        self.retrieval_time = 0
        self.reporting = Reporting()

    def _ensure_server(self, hypers: HypersBase) -> None:
        if hypers.corpus_endpoint.startswith('http'):
            return  # no one starts a corpus service
        if hypers.world_size == 1 and not (hasattr(hypers, 'no_local_dpr_index') and hypers.no_local_dpr_index):
            self.dpr_index = DPRIndex(hypers.corpus_endpoint, dtype=np.float32)
            return  # only one process; no need for a service, we'll use the DPRIndex directly
        # we will start a service ourselves for all our processes to share
        port = hypers.port if hasattr(hypers, 'port') else 5001
        if not hasattr(hypers, 'global_rank') or hypers.global_rank == 0:
            child_pid = os.fork()
            if child_pid == 0:
                from corpus.corpus_server_direct import Options as FlaskOptions, run
                fopts = FlaskOptions()
                fopts.corpus_dir = hypers.corpus_endpoint
                fopts.port = port
                fopts.local_only = hypers.world_size <= torch.cuda.device_count()
                run(fopts)
                exit(0)
            hypers._server_pid = child_pid
        addr = os.environ['MASTER_ADDR'] if 'MASTER_ADDR' in os.environ else 'localhost'
        hypers.corpus_endpoint = f'http://{addr}:{port}'
        if not hasattr(hypers, 'global_rank') or hypers.global_rank == 0:
            # wait until server starts
            headers = {'Content-Type': 'application/json'}
            import requests
            while True:
                time.sleep(5)
                try:
                    test_config = requests.get(hypers.corpus_endpoint + '/config', headers=headers).json()
                    logger.warning(f'Started server: {test_config}')
                    break
                except:
                    logger.warning(f'Waiting on corpus server to start')
                    continue
        if hypers.world_size > 1:
            torch.distributed.barrier()

    @staticmethod
    def cleanup_corpus_server(hypers: HypersBase):
        if hasattr(hypers, '_server_pid'):
            server_pid = hypers._server_pid
        else:
            server_pid = None
        if server_pid is None or server_pid < 0:
            # no corpus server was started
            return
        if hypers.world_size > 1:
            torch.distributed.barrier()  # wait for everyone to finish before killing the corpus server
        if server_pid > 0:
            os.kill(server_pid, signal.SIGKILL)

    def track_retrieval_metrics(self, positive_pids, docs):
        if positive_pids is None:
            return
        assert len(positive_pids) == len(docs)
        pids = [dd['pid'] for dd in docs]
        hit1 = 0
        inrecall = 0
        count = 0
        for positives, retrieved in zip(positive_pids, pids):
            if not positives:
                continue
            if retrieved[0] in positives:
                hit1 += 1
            if any([r in positives for r in retrieved]):
                inrecall += 1
            count += 1
        self.reporting.moving_averages(hit_1=hit1 / count, in_recall=inrecall / count)
        if self.reporting.is_time():
            self.reporting.display()

    def fetch(self, pids, *, include_vectors=True, dummy_if_missing=True):
        start_time = time.time()

        if self.dpr_index is not None:
            rdocs = self.dpr_index.fetch_docs(pids, include_vectors=include_vectors, dummy_if_missing=dummy_if_missing)
        else:
            query = {'pids': pids, 'include_vectors': include_vectors, 'dummy_if_missing': dummy_if_missing}
            response = requests.post(self.endpoint + '/fetch', data=json.dumps(query), headers=self.headers)
            try:
                rdocs = response.json()
            except json.decoder.JSONDecodeError:
                logger.error(f'Bad response: {response}\nRequest: {json.dumps(query)}')
                raise json.decoder.JSONDecodeError

        self.retrieval_time += time.time() - start_time
        docs = rdocs['docs']
        if include_vectors:
            if self.dpr_index is None:
                doc_vectors = np.frombuffer(base64.decodebytes(rdocs['doc_vectors'].encode('ascii')), dtype=self.rest_dtype).\
                    reshape(len(pids), len(pids[0]), -1)
            else:
                doc_vectors = rdocs['doc_vectors']
            retrieved_doc_embeds = torch.Tensor(doc_vectors.copy())
        else:
            retrieved_doc_embeds = None
        return docs, retrieved_doc_embeds

    def retrieve(self, query_vectors: torch.Tensor, *,
                 n_docs=5, n_docs_for_provenance=-1,
                 only_docs=False,
                 get_random=False,
                 gold_pids: List[List[str]] = None,
                 exclude_by_pid_prefix: List[str] = None):
        """
        :param query_vectors: the vectors for the queries, Tensor (batch x dim)
        :param n_docs: dimensionality for doc_scores and doc_vectors
        :param n_docs_for_provenance: number of docs in the returned document list (per query)
        :param only_docs: return only the docs - the doc_scores and doc_vectors will be None
        :param get_random: instead, just get random documents
        :param gold_pids: include these gold passage ids as the first results
        :param exclude_by_pid_prefix: don't include passages that have pids that start with this prefix
        :return: doc_scores, Tensor (batch x n_docs)
                 docs, list of dict [{title: [t_1...t_n_docs_for_provenance], text: [...], pid: [...]}] * batch_size
                 doc_vectors, Tensor (batch x n_docs x dim)
        """
        question_encoder_last_hidden_state = query_vectors  # hidden states of question encoder
        if n_docs_for_provenance < n_docs:
            n_docs_for_provenance = n_docs
        if get_random:
            query = {'get_random': True, 'batch_size': query_vectors.shape[0], 'k': n_docs_for_provenance,
                     'include_vectors': not only_docs}
        else:
            qvecs = question_encoder_last_hidden_state.detach().cpu().numpy()
            if self.dpr_index is None:
                qvecs = base64.b64encode(qvecs.astype(self.rest_dtype)).decode('ascii')
            query = {'query_vectors': qvecs, 'k': n_docs_for_provenance, 'include_vectors': not only_docs}
        if exclude_by_pid_prefix is not None:
            query['exclude_by_pid_prefix'] = exclude_by_pid_prefix
            assert len(exclude_by_pid_prefix) == len(question_encoder_last_hidden_state)
        if gold_pids is not None:
            assert len(gold_pids) == query_vectors.shape[0]
            assert all([isinstance(gp, list) for gp in gold_pids])
            query['gold_pids'] = [gps[:n_docs] for gps in gold_pids]
        start_time = time.time()

        if self.dpr_index is not None:
            rdocs = self.dpr_index.retrieve_docs(query_vectors=query.get('query_vectors', None), k=query['k'],
                                                 batch_size=query.get('batch_size', -1),
                                                 exclude_by_pid_prefix=query.get('exclude_by_pid_prefix', None),
                                                 gold_pids=query.get('gold_pids', None),
                                                 get_random=get_random, include_vectors=query['include_vectors'])
        else:
            response = requests.post(self.endpoint+'/retrieve', data=json.dumps(query), headers=self.headers)
            try:
                rdocs = response.json()
            except json.decoder.JSONDecodeError:
                query['query_vectors'] = ''
                logger.error(f'Bad response: {response}\nRequest: {json.dumps(query)}')
                raise json.decoder.JSONDecodeError

        self.retrieval_time += time.time() - start_time
        docs = rdocs['docs']
        if not only_docs:
            if self.dpr_index is None:
                doc_vectors = np.frombuffer(base64.decodebytes(rdocs['doc_vectors'].encode('ascii')), dtype=self.rest_dtype).\
                    reshape(-1, n_docs_for_provenance, question_encoder_last_hidden_state.shape[-1])
            else:
                doc_vectors = rdocs['doc_vectors']
            doc_vectors = doc_vectors[:, 0:n_docs, :]
            retrieved_doc_embeds = torch.Tensor(doc_vectors.copy()).to(question_encoder_last_hidden_state)
            doc_scores = torch.bmm(
                question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
            ).squeeze(1)
        else:
            doc_scores = None
            retrieved_doc_embeds = None

        return doc_scores, docs, retrieved_doc_embeds
