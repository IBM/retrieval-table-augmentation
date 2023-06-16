import torch
from typing import List, Tuple, Union, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class CandidateTokenizer:
    def __init__(self, tokenizer, answer_normalizer: Callable[[str], str]):
        self.tokenizer = tokenizer
        self.answer_normalizer = answer_normalizer
        # TODO: pass hyperparameters
        self.candidate_start = '«'
        self.candidate_end = '»'
        self.removed_candidate_start = '<'  # if we truncate in the middle of a candidate we replace the last candidate start with this
        self.removed_candidate_end = '>'
        self.title_text_sep = ' | '

        candidate_ndxs = tokenizer(self.candidate_start + self.candidate_end + self.removed_candidate_start,
                                   add_special_tokens=False)['input_ids']
        assert len(candidate_ndxs) == 3
        self.candidate_start_ndx = candidate_ndxs[0]
        self.candidate_end_ndx = candidate_ndxs[1]
        self.removed_candidate_start_ndx = candidate_ndxs[2]

    def ensure_no_candidate_markers(self, text: str):
        return text.replace(self.candidate_start, self.removed_candidate_start).replace(self.candidate_end, self.removed_candidate_end)

    def find_candidate_positions(self, input_ids: torch.LongTensor):
        cand_starts = (input_ids == self.candidate_start_ndx).nonzero(as_tuple=True)
        cand_ends = (input_ids == self.candidate_end_ndx).nonzero(as_tuple=True)
        assert all(cand_starts[0] == cand_ends[0])
        assert cand_starts[0].shape == cand_starts[1].shape
        assert cand_ends[0].shape == cand_ends[1].shape
        assert len(cand_starts[0].shape) == 1 and len(cand_starts[1].shape) == 1
        assert len(cand_starts) == 2
        assert len(cand_ends) == 2
        return cand_starts, cand_ends

    def _remove_truncated_candidates(self, input_tensors):
        input_ids = input_tensors['input_ids']
        for sndx in range(input_ids.shape[0]):
            for tndx in range(input_ids.shape[1]-1, 0, -1):
                if input_ids[sndx, tndx] == self.candidate_end_ndx:
                    # last candidate marker is an end marker - good
                    break
                elif input_ids[sndx, tndx] == self.candidate_start_ndx:
                    # we truncated in the middle of a candidate, change the  start marker to a dummy marker
                    input_ids[sndx, tndx] = self.removed_candidate_start_ndx
                    break

    def _candidate_strings(self, context: str) -> List[str]:
        prev_end = 0
        cand_strings = []
        while True:
            try:
                next_start = context.index(self.candidate_start, prev_end)
                assert next_start >= prev_end
                next_end = context.index(self.candidate_end, next_start)
                assert next_end > next_start
                cand_strings.append(context[next_start+1:next_end])
                prev_end = next_end
            except ValueError:
                break
        return cand_strings

    def _strings_for_final_candidates(self, cand_doc_ndxs: torch.LongTensor, contexts: List[str]):
        cand_strings = []
        prev_end = 0  # also cur_start
        cur_end = 1
        while cur_end <= len(cand_doc_ndxs):
            # if we are at the end or switching to a new document
            if cur_end == len(cand_doc_ndxs) or cand_doc_ndxs[cur_end] != cand_doc_ndxs[prev_end]:
                doc_ndx = cand_doc_ndxs[prev_end]
                doc_cands = self._candidate_strings(contexts[doc_ndx])[:(cur_end-prev_end)]
                cand_strings.extend(doc_cands)
                prev_end = cur_end
            cur_end += 1
        """
        if len(cand_doc_ndxs) != len(cand_strings):
            print(cand_doc_ndxs)
            print(cand_strings)
            print(contexts)
            print([self._candidate_strings(context) for context in contexts])
        """
        assert len(cand_doc_ndxs) == len(cand_strings)
        return cand_strings

    def tokenize_with_candidates(self,
                                 query: Union[str, Tuple[str, str]],
                                 answers: Optional[List[str]],
                                 docs: List[dict],
                                 n_docs: int, max_seq_length: int):
        contexts = [title + self.title_text_sep + text
                    for title, text in zip(docs[0]['title'][:n_docs], docs[0]['text'][:n_docs])]
        assert len(contexts) == n_docs

        if type(query) == tuple:
            assert len(query) == 2 and all(type(q) == str for q in query)
            query = query[0] + self.title_text_sep + query[1]
        assert type(query) == str
        query = self.ensure_no_candidate_markers(query)  # make sure we don't have any candidate markers in query

        input_tensors = self.tokenizer(
            [query] * len(contexts),
            contexts,
            max_length=max_seq_length,
            truncation=True, padding=True,
            return_tensors='pt'
        )

        self._remove_truncated_candidates(input_tensors)
        cand_starts, cand_ends = self.find_candidate_positions(input_tensors['input_ids'])
        cand_strings = self._strings_for_final_candidates(cand_starts[0], contexts)
        if answers is None:
            answers = []
        cand_strings = [self.answer_normalizer(cs) for cs in cand_strings]
        cand_correct = [cs in answers for cs in cand_strings]
        return input_tensors, cand_starts, cand_ends, cand_correct, cand_strings