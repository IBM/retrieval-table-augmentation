from dataloader.distloader_base import MultiFileLoader, DistBatchesBase
from midpr.biencoder_hypers import BiEncoderHypers
import ujson as json
from typing import List, Tuple
from transformers import PreTrainedTokenizerFast
import torch
import logging
import random

logger = logging.getLogger(__name__)


class BiEncoderInst:
    __slots__ = 'qry_title', 'qry_text', 'pos_ctx', 'neg_ctx'

    def __init__(self, qry_title: str, qry_text: str, pos_ctx: List[Tuple[str, str]], neg_ctx: List[Tuple[str, str]]):
        self.qry_title = qry_title
        self.qry_text = qry_text
        self.pos_ctx = pos_ctx
        self.neg_ctx = neg_ctx


class BiEncoderBatches(DistBatchesBase):
    def __init__(self, insts: List[BiEncoderInst], hypers: BiEncoderHypers,
                 qry_tokenizer: PreTrainedTokenizerFast, ctx_tokenizer: PreTrainedTokenizerFast):
        super().__init__(insts, hypers)
        self.hypers = hypers
        self.qry_tokenizer = qry_tokenizer
        self.ctx_tokenizer = ctx_tokenizer

    def make_batch(self, index, insts):
        ctx_titles = [c[0] for i in insts for c in i.pos_ctx + i.neg_ctx]
        ctx_texts = [c[1] for i in insts for c in i.pos_ctx + i.neg_ctx]
        positive_mask = torch.zeros((len(insts), len(ctx_titles)), dtype=torch.long)
        ctx_offset = 0
        for qi, i in enumerate(insts):
            num_pos = len(i.pos_ctx)
            positive_mask[qi, ctx_offset:ctx_offset+num_pos] = 1
            ctx_offset += num_pos + len(i.neg_ctx)

        ctxs_tensors = self.ctx_tokenizer(ctx_titles, ctx_texts, max_length=self.hypers.seq_len_c,
                                          truncation=True, padding="longest", return_tensors="pt")
        if not self.hypers.query_single_sequence:
            qry_titles = [i.qry_title for i in insts]
            qry_texts = [i.qry_text for i in insts]
            qrys_tensors = self.qry_tokenizer(qry_titles, qry_texts, max_length=self.hypers.seq_len_q,
                                              truncation=True, padding="longest", return_tensors="pt")
        else:
            qrys = [i.qry_title + ' | ' + i.qry_text if i.qry_title else i.qry_text for i in insts]
            qrys_tensors = self.qry_tokenizer(qrys, max_length=self.hypers.seq_len_q,
                                              truncation=True, padding="longest", return_tensors="pt")
        return qrys_tensors['input_ids'], qrys_tensors['attention_mask'], \
               ctxs_tensors['input_ids'], ctxs_tensors['attention_mask'], \
               positive_mask


class BiEncoderLoader(MultiFileLoader):
    def __init__(self, hypers: BiEncoderHypers, per_gpu_batch_size: int, qry_tokenizer, ctx_tokenizer, data_dir, *,
                 files_per_dataloader=1, checkpoint_info=None):
        super().__init__(hypers, per_gpu_batch_size, data_dir,
                         checkpoint_info=checkpoint_info, files_per_dataloader=files_per_dataloader)
        self.hypers = hypers
        self.qry_tokenizer = qry_tokenizer
        self.ctx_tokenizer = ctx_tokenizer

    def batch_dict(self, batch):
        """
        :param batch: input_ids_q, attention_mask_q, input_ids_c, attention_mask_c, positive_indices
        :return:
        """
        batch = tuple(t.to(self.hypers.device) for t in batch)
        return {'input_ids_q': batch[0], 'attention_mask_q': batch[1],
                'input_ids_c': batch[2], 'attention_mask_c': batch[3],
                'positive_mask': batch[4]}

    def display_batch(self, batch):
        input_ids_q = batch[0]
        input_ids_c = batch[2]
        positive_mask = batch[4]
        logger.info(f'{input_ids_q.shape} queries and {input_ids_c.shape} contexts')
        qndx = random.randint(0, input_ids_q.shape[0]-1)
        logger.info(f'  query: {self.qry_tokenizer.decode(input_ids_q[qndx])}')
        cndx = random.randint(0, input_ids_c.shape[0]-1)
        logger.info(f'passage: {self.ctx_tokenizer.decode(input_ids_c[cndx])}')

    def _one_load(self, lines):
        insts = []
        for line in lines:
            jobj = json.loads(line)
            if isinstance(jobj['query'], str):
                qry_title = ''
                qry_text = jobj['query']
                assert self.hypers.query_single_sequence
            else:
                qry_title = jobj['query']['title']
                qry_text =  jobj['query']['text']
            json_pos = jobj['positives']
            json_negs = jobj['negatives']
            positives = [(pos['title'], pos['text']) for pos in json_pos]
            if len(json_negs) == 0:
                logger.warning(f'bad instance! {len(json_negs)} negatives')
                continue
            if self.hypers.sample_negative_from_top_k > 0:
                neg_ndx = random.randint(0, min(len(json_negs), self.hypers.sample_negative_from_top_k)-1)
                negatives = [json_negs[neg_ndx]['title'], json_negs[neg_ndx]['text']]
            else:
                negatives = [(neg['title'], neg['text']) for neg in json_negs]
            insts.append(BiEncoderInst(qry_title, qry_text, positives, negatives))
        return BiEncoderBatches(insts, self.hypers, self.qry_tokenizer, self.ctx_tokenizer)
