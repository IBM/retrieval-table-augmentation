import torch
from typing import List, Tuple, Optional, Mapping
from transformers import PreTrainedTokenizerFast


def tokenize_with_candidate_answers(tokenizer: PreTrainedTokenizerFast,
                          question: str, context_batch: List[str],
                          answers: Optional[List[str]], max_seq_len: int, *,
                          candidate_start: str = '«', candidate_end: str = '»', case_insensitive: bool = False) -> \
        Tuple[Mapping[str, torch.LongTensor], List[Tuple[List[int], List[int]]]]:
    if answers:
        if case_insensitive:
            answers = [candidate_start + ans.lower() + candidate_end for ans in answers]
            answer_spans_batch = [find_answer_spans(context.lower(), answers) for context in context_batch]
        else:
            answers = [candidate_start + ans + candidate_end for ans in answers]
            answer_spans_batch = [find_answer_spans(context, answers) for context in context_batch]
    else:
        answer_spans_batch = [] * len(context_batch)
    question_batch = [question] * len(context_batch)
    return tokenize_with_answer_spans(tokenizer, question_batch, context_batch, answer_spans_batch, max_seq_len)


def tokenize_with_answers(tokenizer: PreTrainedTokenizerFast,
                          question_batch: List[str], context_batch: List[str],
                          answers_batch: List[List[str]], max_seq_len: int):
    answer_spans_batch = [find_answer_spans(context, answers) for context, answers in zip(context_batch, answers_batch)]
    return tokenize_with_answer_spans(tokenizer, question_batch, context_batch, answer_spans_batch, max_seq_len)


def tokenize_with_answer_spans(tokenizer: PreTrainedTokenizerFast,
                               question_batch: List[str], context_batch: List[str],
                               answer_spans_batch: List[List[Tuple[int, int]]], max_seq_len: int) -> \
        Tuple[Mapping[str, torch.LongTensor], List[Tuple[List[int], List[int]]]]:
    input_tensors = tokenizer(
        question_batch,
        context_batch,
        max_length=max_seq_len,
        truncation=True, padding=True,
        return_offsets_mapping=True,
        return_tensors='pt'
    )
    input_ids = input_tensors['input_ids']
    # attention_mask = input_tensors['attention_mask']
    # token_type_ids = input_tensors['token_type_ids']
    offset_mapping = input_tensors['offset_mapping']
    token_spans_batch = []
    for bi, offsets in enumerate(offset_mapping):
        # find token bounds of context
        sequence_ids = input_tensors.sequence_ids(bi)
        context_start_token = 0
        while sequence_ids[context_start_token] != 1:
            context_start_token += 1
        context_end_token = input_ids.shape[-1] - 1
        while sequence_ids[context_end_token] != 1:
            context_end_token -= 1

        # find correct start and correct end token indices
        starts = []
        ends = []
        for answer_span in answer_spans_batch[bi]:
            start = context_start_token
            end = context_end_token
            while start < len(offsets) and offsets[start][0] <= answer_span[0]:
                start += 1
            while offsets[end][1] >= answer_span[1]:
                end -= 1
            start = start - 1
            end = end + 1
            if start <= end:
                # NOTE: don't add duplicates
                if start not in starts:
                    starts.append(start)
                if end not in ends:
                    ends.append(end)
        token_spans_batch.append((starts, ends))
    return input_tensors, token_spans_batch


def overlaps(span1: Tuple[int, int], span2: Tuple[int, int]):
    s1, e1 = span1
    s2, e2 = span2
    return s1 < e2 and s2 < e1


def find_answer_spans(context: str, answers) -> List[Tuple[int, int]]:
    """
    find the spans in context that correspond to any string in answers
    :param context: string
    :param answers: iterable of strings
    :return: list of tuples [start, end) character spans
    """
    unfiltered_matches = []
    for answer in answers:
        last_match_end = 0
        while True:
            start_loc = context.find(answer, last_match_end)
            if start_loc >= 0:
                end_loc = start_loc + len(answer)
                unfiltered_matches.append((start_loc, end_loc))
                last_match_end = end_loc
            else:
                break
    unfiltered_matches.sort(key=lambda x: x[1] - x[0], reverse=True)
    non_overlapping = []
    for span in unfiltered_matches:
        if len(non_overlapping) == 0 or not any([overlaps(span, fspan) for fspan in non_overlapping]):
            non_overlapping.append(span)
    non_overlapping.sort(key=lambda x: x[0])
    return non_overlapping
