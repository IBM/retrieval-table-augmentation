from midpr.biencoder_hypers import BiEncoderHypers
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import (DPRQuestionEncoder, DPRContextEncoder)
import os
from typing import Union
import logging

logger = logging.getLogger(__name__)


class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask, dummy):
        pooler_output = self.encoder(input_ids, attention_mask)[0]
        return pooler_output


class BiEncoder(torch.nn.Module):
    """
    This trains the DPR encoders to maximize dot product between queries and positive contexts.
    We only use this model during training.
    """
    def __init__(self, hypers: BiEncoderHypers):
        super().__init__()
        self.hypers = hypers
        self.qry_model = EncoderWrapper(DPRQuestionEncoder.from_pretrained(hypers.qry_encoder_name_or_path))
        self.ctx_model = EncoderWrapper(DPRContextEncoder.from_pretrained(hypers.ctx_encoder_name_or_path))
        assert hypers.world_size == 1

    def encode(self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        if 0 < self.hypers.encoder_gpu_train_limit:
            # checkpointing
            # dummy requries_grad to deal with checkpointing issue:
            #   https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/13
            dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
            all_pooled_output = []
            for sub_bndx in range(0, input_ids.shape[0], self.hypers.encoder_gpu_train_limit):
                sub_input_ids = input_ids[sub_bndx:sub_bndx+self.hypers.encoder_gpu_train_limit]
                sub_attention_mask = attention_mask[sub_bndx:sub_bndx + self.hypers.encoder_gpu_train_limit]
                pooler_output = checkpoint(model, sub_input_ids, sub_attention_mask, dummy_tensor)
                all_pooled_output.append(pooler_output)
            return torch.cat(all_pooled_output, dim=0)
        else:
            return model(input_ids, attention_mask, None)

    def forward(
        self,
        input_ids_q: torch.Tensor,
        attention_mask_q: torch.Tensor,
        input_ids_c: torch.Tensor,
        attention_mask_c: torch.Tensor,
        positive_mask: torch.Tensor
    ):
        """
        All batches must be the same size (q and c are fixed during training)
        :param input_ids_q: q x seq_len_q [0, vocab_size)
        :param attention_mask_q: q x seq_len_q [0 or 1]
        :param input_ids_c: c x seq_len_c
        :param attention_mask_c: c x seq_len_c
        :param positive_mask: q x c [0 or 1]
        :return:
        """
        qry_reps = self.encode(self.qry_model, input_ids_q, attention_mask_q)
        ctx_reps = self.encode(self.ctx_model, input_ids_c, attention_mask_c)
        dot_products = torch.matmul(qry_reps, ctx_reps.transpose(0, 1))  # q x c
        probs = F.log_softmax(dot_products, dim=1)
        assert probs.shape == positive_mask.shape
        # we set the non-positive log-probs to very low so they don't affect the logsumexp
        loss = -(torch.logsumexp(probs + (-10000 * (1-positive_mask)), 1).mean())
        predictions = torch.max(probs, 1)[1]
        accuracy = sum([positive_mask[qndx, predictions[qndx]] for qndx in range(len(predictions))]) / len(predictions)
        return loss, accuracy

    def save(self, save_dir: Union[str, os.PathLike]):
        self.qry_model.encoder.save_pretrained(os.path.join(save_dir, 'qry_encoder'))
        self.ctx_model.encoder.save_pretrained(os.path.join(save_dir, 'ctx_encoder'))
