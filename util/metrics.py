from typing import List, Dict, Optional, Mapping, Tuple, Callable
import torch
import math
from collections import defaultdict


def ndcg(correct_ranks: List[int], possible_correct: int, limit: int) -> float:
    """
    Compute and return the NDCG
    :param correct_ranks: the 1-based ranks of the correct predictions
    :param possible_correct: the number of correct or relevant results
    :param limit: for example 10 in NDCG10
    :return: NDCG_{limit}
    """
    idcg = sum(1.0 / math.log(i + 1, 2) for i in range(1, possible_correct + 1) if i <= limit)
    dcg = sum(1.0 / math.log(rank + 1, 2) for rank in correct_ranks if rank <= limit)
    return dcg / idcg


def graded_ndcg(retrieved_relevance: List[float], ideal_retrieved_relevance: List[float], limit: int) -> float:
    """
    Compute and return NDCG@limit
    @param retrieved_relevance:
    @param ideal_retrieved_relevance:
    @param limit:
    @return:
    """
    idcg = sum(relevance / math.log(zrank + 2, 2) for zrank, relevance in enumerate(ideal_retrieved_relevance[:limit]))
    dcg = sum(relevance / math.log(zrank + 2, 2) for zrank, relevance in enumerate(retrieved_relevance[:limit]))
    return dcg / idcg


def reciprocal_rank(correct_ranks: List[int]):
    """
    compute reciprocal rank, suitable for tracking mean reciprocal rank (MRR)
    :param correct_ranks: the 1-based ranks of the correct predictions
    :return: reciprocal rank of first correct prediction
    """
    if not correct_ranks:
        return 0.0
    r = min(correct_ranks)
    assert r >= 1
    return 1.0 / r


def aggregate_by_answer_string(cand_strings: List[str],
                               candidate_logits: torch.FloatTensor,
                               cand_correct: List[bool],
                               max_not_sum: bool = True) -> List[Tuple[str, float, bool]]:
    """
    Aggregate per-occurrence logits to a per-candidate-answer score
    :param cand_strings:
    :param candidate_logits:
    :param cand_correct:
    :param max_not_sum:
    :return: sorted list of answer, score, correct
    """
    cand2score = defaultdict(list)
    cand2correct = dict()
    for cand_string, cand_logit, correct in zip(cand_strings, candidate_logits.numpy().tolist(), cand_correct):
        cand2score[cand_string].append(cand_logit)
        if cand_string in cand2correct:
            assert correct == cand2correct[cand_string]
        else:
            cand2correct[cand_string] = correct

    if max_not_sum:
        agg_func: Callable[[List[float]], float] = lambda scores: max(scores)
    else:
        agg_func: Callable[[List[float]], float] = lambda scores: torch.logsumexp(torch.tensor(scores), dim=0).item()
    aggregated = [(cand, agg_func(scores), cand2correct[cand]) for cand, scores in cand2score.items()]

    aggregated.sort(key=lambda x: x[1], reverse=True)
    return aggregated


def mrr(candidate_logits: torch.FloatTensor, cand_correct: List[bool]):
    score_and_correct = list(zip(candidate_logits, cand_correct))
    score_and_correct.sort(key=lambda x: x[0], reverse=True)
    rr = 0.0
    for zrank, cand_tuple in enumerate(score_and_correct):
        if cand_tuple[1]:
            rr = 1.0 / (zrank + 1)
            break
    return rr


def answer_metrics(cand_strings: List[str], candidate_logits: torch.FloatTensor,
                   cand_correct: List[bool], cand_doc_ndxs: torch.LongTensor, num_answers: int) -> Dict[str, float]:
    """
    Compute metrics during training
    len(candidate_logits) == len(cand_correct) == len(cand_doc_ndxs)
    :param cand_strings: the string candidates
    :param candidate_logits: logit for each occurrence of candidate
    :param cand_correct: whether each candidate occurrence is correct
    :param cand_doc_ndxs: == cand_starts[0] == cand_ends[0], the document indexes for the candidates
    :return: mrr_max_agg, mrr_sum_agg, answer_bearing_mrr
    """
    if sum(cand_correct) == 0:
        return {'mrr_max_agg': 0.0, 'mrr_sum_agg': 0.0, 'ndcg_10': 0.0, 'ndcg_20': 0.0, 'answer_bearing_mrr': 0.0}
    # track answer-bearing mrr
    answer_bearing_rank = -1
    for answer_bearing, doc_ndx in zip(cand_correct, cand_doc_ndxs):
        if answer_bearing:
            answer_bearing_rank = doc_ndx + 1
            break

    # compute MRR aggregated by max
    correct_ranks = []
    for zrank, cand_tuple in enumerate(
            aggregate_by_answer_string(cand_strings, candidate_logits, cand_correct, max_not_sum=True)):
        if cand_tuple[2]:
            correct_ranks.append(zrank + 1)
    mrr_max_agg = 1.0 / correct_ranks[0] if len(correct_ranks) > 0 else 0

    # compute MRR aggregated by sum
    correct_ranks = []
    for zrank, cand_tuple in enumerate(
            aggregate_by_answer_string(cand_strings, candidate_logits, cand_correct, max_not_sum=False)):
        if cand_tuple[2]:
            correct_ranks.append(zrank + 1)
    mrr_sum_agg = 1.0 / correct_ranks[0] if len(correct_ranks) > 0 else 0

    # NDCG based on aggregation by sum
    ndcg_10 = ndcg(correct_ranks, num_answers, 10)
    ndcg_20 = ndcg(correct_ranks, num_answers, 20)

    return {'mrr_max_agg': mrr_max_agg, 'mrr_sum_agg': mrr_sum_agg,
            'ndcg_10': ndcg_10, 'ndcg_20': ndcg_20,
            'answer_bearing_mrr': 1.0 / answer_bearing_rank if answer_bearing_rank > 0 else 0}