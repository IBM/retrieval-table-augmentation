from util.line_corpus import jsonl_lines, block_shuffle
import ujson as json
import tabulate
from table_augmentation.explore.table_passages import table2query
from dpr.retriever_dpr import DPRHypers, RetrieverDPR
from torch_util.hypers_base import HypersBase
from dpr.retriever_bm25 import BM25Hypers, RetrieverBM25
from extractive.extractive_util import find_answer_spans


class Options(HypersBase):
    def __init__(self):
        super().__init__()
        self.queries = ''
        self.anserini_jar = ''
        self.anserini_index = ''
        self.dpr = DPRHypers()
        self.bm25 = BM25Hypers()
        self.__required_args__ = ['dpr.corpus_endpoint', 'bm25.jar', 'bm25.anserini_index', 'queries']

    def _post_init(self):
        self._quiet_post_init = True
        super()._post_init()
        self.dpr.copy_from_base_hypers(self, 1, per_gpu_batch_size_scale=1)


def highlight(text, spans):
    starts = [(s, 's') for s, e in spans]
    ends = [(e, 'e') for s, e in spans]
    points = starts + ends
    points.sort(key=lambda x:x[0])
    highlighted = ''
    prev_ndx = 0
    for ndx, type in points:
        highlighted = highlighted + text[prev_ndx:ndx] + ('<<' if type == 's' else '>>')
        prev_ndx = ndx
    return highlighted + text[prev_ndx:]


def show_results(doc_scores, docs, answers):
    highlighted = [score + ': ' + title + '\n\n' + highlight(text, find_answer_spans(text, answers))
                   for score, title, text in zip(doc_scores, docs['title'], docs['text'])]
    for r in highlighted:
        print(r)


if __name__ == "__main__":
    # sample a table
    #   show how we can make queries out of it
    #   show how we make passages out of it
    #   query our index, highlight spans in result passages that are correct answers
    opts = Options()
    opts.fill_from_args()

    dpr = RetrieverDPR(opts.dpr)
    bm25 = RetrieverBM25(opts.bm25)

    for ndx, line in enumerate(block_shuffle(jsonl_lines(opts.queries, shuffled=True))):
        jobj = json.loads(line)
        title = jobj['title']
        header = jobj['header']
        rows = jobj['rows']
        print('\n' + '=' * 80)
        print(title)
        print(tabulate.tabulate(rows, header))
        query = table2query(jobj, num_seeds=2, row_or_column='r')
        query_str = query['title'] + '\n\n' + query['text']
        answers = query['answers']
        print(f'Query:\n{query_str}')
        print(f'Answers:\n{answers}')
        dpr_doc_scores, dpr_docs, _ = dpr.retrieve_forward([query_str], exclude_by_pid_prefix=jobj['table_id'])
        bm25_doc_scores, bm25_docs = bm25.retrieve_forward([query_str], exclude_by_pid_prefix=jobj['table_id'])
        print('DPR Results')
        show_results(dpr_doc_scores, dpr_docs, answers)
        print('BM25 Results')
        show_results(bm25_doc_scores, bm25_docs, answers)
        print('='*80)
        if ndx >= 10:
            break


"""

"""