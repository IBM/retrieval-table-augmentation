import logging
import jnius_config
import multiprocessing
from multiprocessing.pool import ThreadPool
import functools
from typing import List, Union, Tuple, Optional, Set, Dict
import ujson as json

logger = logging.getLogger(__name__)


class BM25Hypers:
    def __init__(self):
        super().__init__()
        self.jar = ''
        self.anserini_index = ''
        self.n_docs = 10
        self.allow_fewer_results = False
        self.rm3 = False
        self.num_processes = multiprocessing.cpu_count()


def _retrieve_one(query_exclude: Tuple[str, Set[str], Union[bool, str]], searcher, hypers: BM25Hypers, JString) -> \
            Tuple[List[float], Dict[str, List[str]]]:
    global _DISPLAYED_ONE_RAW_HITS
    query, exclude, exclude_pid_prefix = query_exclude
    initial_n = (30 + 3 * hypers.n_docs + len(exclude)) if exclude_pid_prefix else (2 * hypers.n_docs + len(exclude))
    title_end_text_start = []
    if JString is not None:
        hits = searcher.search(JString(query.encode('utf-8')), initial_n)
    else:
        hits = searcher.search(query, initial_n)
        for hit in hits:
            raw = json.loads(hit.raw)
            if 'title_text_sep' in raw:
                title_end, sep_len = raw['title_text_sep']
                title_end_text_start.append((title_end, title_end + sep_len))
            hit.content = raw['contents']
    if not title_end_text_start:
        title_lens = [hit.content.find(RetrieverBM25.TITLE_TEXT_SEP) for hit in hits]
        title_end_text_start = [(tl, tl+len(RetrieverBM25.TITLE_TEXT_SEP)) for tl in title_lens]
    else:
        assert len(title_end_text_start) == len(hits)
    hits = [hit for hit in hits if
            hit.content not in exclude and
            (not exclude_pid_prefix or not (hit.docid+'\n').startswith(exclude_pid_prefix))][:hypers.n_docs]
    if not hypers.allow_fewer_results:
        if len(hits) == 0:
            # create dummy docs if no result
            doc_scores = [0.0] * hypers.n_docs
            docs = {'pid': ['N/A:0'] * hypers.n_docs,
                    'title': ['title'] * hypers.n_docs,
                    'text': ['text'] * hypers.n_docs}
            logger.warning(f'No results for {query}!')
            return doc_scores, docs
        if len(hits) < hypers.n_docs:
            # duplicate last doc if too few results
            logger.warning(f'Too few results for {query}! ({len(hits)})')
            title_end_text_start.extend(title_end_text_start[-1] * (hypers.n_docs - len(hits)))
            hits.extend([hits[-1]] * (hypers.n_docs - len(hits)))
        assert len(hits) == hypers.n_docs
    doc_scores = [hit.score for hit in hits]
    titles = [hit.content[:title_text[0]] for hit, title_text in zip(hits, title_end_text_start)]
    texts = [hit.content[title_text[1]:] for hit, title_text in zip(hits, title_end_text_start)]
    docs = {'pid': [hit.docid for hit in hits], 'title': titles, 'text': texts}
    return doc_scores, docs


class RetrieverBM25:
    TITLE_TEXT_SEP = '\n\n'

    def __init__(self, hypers: BM25Hypers):
        """
        :param hypers:
        """
        self.hypers = hypers
        if hypers.jar:
            jnius_config.set_classpath(hypers.jar)
            from jnius import autoclass
            self.JString = autoclass('java.lang.String')
            JSearcher = autoclass('io.anserini.search.SimpleSearcher')
            self.searcher = JSearcher(self.JString(hypers.anserini_index))
            if hypers.rm3:  # NOTE: no positive impact seen so far
                self.searcher.setRM3Reranker()  # NOTE: this is just setRM3() in later versions of Anserini
        else:
            from pyserini.search import SimpleSearcher
            self.searcher = SimpleSearcher(hypers.anserini_index)
            self.JString = None  # we don't use this with pyserini
            if hypers.rm3:
                self.searcher.set_rm3()

        if hypers.num_processes > 1:
            # NOTE: only thread-based pooling works with the JSearcher
            self.pool = ThreadPool(processes=hypers.num_processes)
            logger.info(f'Using multiprocessing pool with {hypers.num_processes} workers')
        else:
            self.pool = None
        self._retrieve_one = functools.partial(_retrieve_one,
                                               searcher=self.searcher, hypers=self.hypers, JString=self.JString)

    def close(self):
        try:
            self.searcher.close()
        except:
            pass

    def retrieve_forward(self, queries: Union[List[str], List[Tuple[str, str]]], *,
                         exclude_by_content: Optional[List[Set[str]]] = None,
                         exclude_by_pid_prefix: Optional[List[Union[str, bool]]] = None) -> \
            Tuple[List[List[float]], List[Dict[str, List[str]]]]:
        """

        :param queries: list of queries to retrieve documents for
        :param exclude_by_content: exclude results that have the specified content
        :param exclude_by_pid_prefix: exclude results that have pids starting with the prefix
        :return: doc_scores, docs
        """
        if type(queries[0]) == tuple:
            assert all(type(q) == tuple and len(q) == 2 and type(q[0]) == str and type(q[1]) == str for q in queries)
            queries = [q[0] + RetrieverBM25.TITLE_TEXT_SEP + q[1] for q in queries]
        assert all(type(q) == str for q in queries)
        if exclude_by_content is None:
            exclude_by_content = [set() for _ in range(len(queries))]
        else:
            assert len(exclude_by_content) == len(queries)
        if exclude_by_pid_prefix is None:
            exclude_by_pid_prefix = [False for _ in range(len(queries))]

        if self.pool is not None:
            result_batch = self.pool.map(self._retrieve_one, zip(queries, exclude_by_content, exclude_by_pid_prefix))
            docs = [r[1] for r in result_batch]
            doc_scores = [r[0] for r in result_batch]
        else:
            docs = []
            doc_scores = []
            for query, exclude, exclude_pid_prefix in zip(queries, exclude_by_content, exclude_by_pid_prefix):
                doc_scores_i, docs_i = self._retrieve_one((query, exclude, exclude_pid_prefix))
                doc_scores.append(doc_scores_i)
                docs.append(docs_i)

        return doc_scores, docs
