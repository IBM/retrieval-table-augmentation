import numpy as np
import os
import logging
from dpr.simple_mmap_dataset import Corpus
from dpr.faiss_index import ANNIndex
import random
from typing import List, Optional

logger = logging.getLogger(__name__)


class DPRIndex:
    def __init__(self, corpus_dir: str, dtype=np.float16):
        self.corpus_dir = corpus_dir
        # we either have a single index.faiss or we have an index for each offsets/passages
        if os.path.exists(os.path.join(self.corpus_dir, "index.faiss")):
            self.passages = Corpus(os.path.join(self.corpus_dir))
            self.index = ANNIndex(os.path.join(self.corpus_dir, "index.faiss"))
            self.shards = None
            self.dim = self.index.dim()
        else:
            self.shards = []
            # loop over the different index*.faiss
            # so we have a list of (index, passages)
            # we search each index, then take the top-k results overall
            for filename in os.listdir(self.corpus_dir):
                if filename.startswith('passages') and filename.endswith('.json.gz.records'):
                    name = filename[len("passages"):-len(".json.gz.records")]
                    self.shards.append((ANNIndex(os.path.join(self.corpus_dir, f'index{name}.faiss')),
                                       Corpus(os.path.join(self.corpus_dir, f'passages{name}.json.gz.records'))))
            self.dim = self.shards[0][0].dim()
            assert all([self.dim == shard[0].dim() for shard in self.shards])
            logger.info(f'Using sharded faiss with {len(self.shards)} shards.')
        logger.info(f'DPR Index dimension = {self.dim}')
        self.dummy_doc = {'pid': 'N/A', 'title': '', 'text': '', 'vector': np.zeros(self.dim, dtype=dtype)}
        self.dtype = dtype

    def get_config(self):
        return {'dtype': 16 if self.dtype == np.float16 else 32, 'dim': self.dim, 'corpus': self.corpus_dir}

    def merge_results(self, query_vectors, k):
        # CONSIDER: consider ResultHeap (https://github.com/matsui528/faiss_tips)
        all_scores = np.zeros((query_vectors.shape[0], k * len(self.shards)), dtype=np.float32)
        all_indices = np.zeros((query_vectors.shape[0], k * len(self.shards)), dtype=np.int64)
        for si, shard in enumerate(self.shards):
            index_i, passages_i = shard
            scores, indexes = index_i.search(query_vectors, k)
            assert len(scores.shape) == 2
            assert scores.shape[1] == k
            assert scores.shape == indexes.shape
            assert scores.dtype == np.float32
            assert indexes.dtype == np.int64
            all_scores[:, si * k: (si + 1) * k] = scores
            all_indices[:, si * k: (si + 1) * k] = indexes
        kbest = all_scores.argsort()[:, -k:][:, ::-1]
        docs = [[self.shards[ndx // k][1][all_indices[bi, ndx]] for ndx in ndxs] for bi, ndxs in enumerate(kbest)]
        return docs

    def _random_docs(self, batch_size: int, k: int):
        num_passages = sum([len(s[1]) for s in self.shards]) if self.shards is not None else len(self.passages)

        def get_random():
            ndx = random.randint(0, num_passages-1)
            if self.shards is None:
                return self.passages[ndx]
            offset = 0
            for si in range(len(self.shards)):
                if ndx - offset < len(self.shards[si][1]):
                    return self.shards[si][1][ndx - offset]
                offset += len(self.shards[si][1])
            raise ValueError

        return [[get_random() for _ in range(k)] for _ in range(batch_size)]

    def _get_docs_by_pids(self, pids: List[str], *, dummy_if_missing=False):
        docs = []
        for pid in pids:
            doc = None
            if self.shards is None:
                doc = self.passages.get_by_pid(pid)
            else:
                for shard in self.shards:
                    doc = shard[1].get_by_pid(pid)
                    if doc is not None:
                        break
            if doc is None:
                if dummy_if_missing:
                    doc = self.dummy_doc
                else:
                    raise ValueError
            docs.append(doc)
        return docs

    def fetch_docs(self, pids: List[List[str]], include_vectors: bool = False, dummy_if_missing: bool = False):
        # input is 'pids': list of list of ids to get
        # and boolean for include vectors
        batch_size = len(pids)
        k = len(pids[0])
        docs = [self._get_docs_by_pids(pids, dummy_if_missing=dummy_if_missing) for pids in pids]
        assert all([len(d) == k for d in docs])
        doc_dicts = [{'pid': [dqk['pid'] for dqk in dq],
                      'title': [dqk['title'] for dqk in dq],
                      'text': [dqk['text'] for dqk in dq]} for dq in docs]

        retval = {'docs': doc_dicts}
        if include_vectors:
            doc_vectors = np.zeros([batch_size, k, self.dim], dtype=self.dtype)
            for qi, docs_qi in enumerate(docs):
                for ki, doc_qi_ki in enumerate(docs_qi):
                    doc_vectors[qi, ki] = doc_qi_ki['vector']
            retval['doc_vectors'] = doc_vectors

        return retval

    def retrieve_docs(self, query_vectors: Optional[np.ndarray], k: int,
                      exclude_by_pid_prefix: Optional[List[str]] = None,
                      gold_pids: Optional[List[List[str]]] = None,
                      batch_size: int = -1, get_random: bool = False, include_vectors: bool=False):
        # input is three parts:
        #  the base64 encoded fp16 numpy matrix
        #  k (the number of records per document)
        #  return-vectors flag

        if get_random:
            docs = self._random_docs(batch_size, k)
        else:
            query_vectors = query_vectors.astype(np.float32)
            batch_size = query_vectors.shape[0]
            assert query_vectors.shape[1] == self.dim
            if exclude_by_pid_prefix is not None:
                assert len(exclude_by_pid_prefix) == batch_size and all(type(pp) == str for pp in exclude_by_pid_prefix)
                initial_k = k + 10
                docs = []
                while len(docs) == 0 or any([len(dq) < k for dq in docs]):
                    initial_k *= 2
                    if self.shards is None:
                        scores, indexes = self.index.search(query_vectors, initial_k)
                        docs = [[self.passages[ndx] for ndx in ndxs] for ndxs in indexes]
                    else:
                        docs = self.merge_results(query_vectors, initial_k)
                    docs = [[dqk for dqk in dq if not (dqk['pid'] + '\n').startswith(exclude_by_pid_prefix[qi])][:k]
                            for qi, dq in enumerate(docs)]
            else:
                if self.shards is None:
                    scores, indexes = self.index.search(query_vectors, k)
                    docs = [[self.passages[ndx] for ndx in ndxs] for ndxs in indexes]
                else:
                    docs = self.merge_results(query_vectors, k)

        # add the gold_pids to the docs if requested
        if gold_pids is not None:
            assert len(gold_pids) == batch_size
            gdocs = []
            for qi in range(batch_size):
                gpids = gold_pids[qi][:k]
                assert isinstance(gpids, list)
                gold_docs = self._get_docs_by_pids(gpids)
                gdocs.append(gold_docs + [dqk for dqk in docs[qi] if dqk['pid'] not in gpids][:k-len(gold_docs)])
            docs = gdocs
            assert all([len(d) == k for d in docs])

        if 'pid' in docs[0][0]:
            doc_dicts = [{'pid': [dqk['pid'] for dqk in dq],
                          'title': [dqk['title'] for dqk in dq],
                          'text': [dqk['text'] for dqk in dq]} for dq in docs]
        else:
            doc_dicts = [{'title': [dqk['title'] for dqk in dq],
                          'text': [dqk['text'] for dqk in dq]} for dq in docs]

        retval = {'docs': doc_dicts}
        if include_vectors:
            doc_vectors = np.zeros([batch_size, k, self.dim], dtype=self.dtype)
            if not get_random:
                for qi, docs_qi in enumerate(docs):
                    if gold_pids is not None:
                        gpids = gold_pids[qi]
                    else:
                        gpids = []
                    for ki, doc_qi_ki in enumerate(docs_qi):
                        # if we have gold_pids, set their vector to 100 * the query vector
                        if ki < len(gpids):
                            doc_vectors[qi, ki] = 100 * query_vectors[qi]
                        else:
                            doc_vectors[qi, ki] = doc_qi_ki['vector']
            retval['doc_vectors'] = doc_vectors
        # print(retval)
        # output
        #   list of docs: len(docs) == query_vectors.shape[0]; len(docs[i].title) == len(docs[i].text) == k
        #   doc_vectors: query_vectors.shape[0] x k x query_vectors.shape[1]
        return retval
