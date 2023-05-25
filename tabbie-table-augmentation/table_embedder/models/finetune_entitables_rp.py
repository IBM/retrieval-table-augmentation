from typing import Dict, Optional, Tuple
from overrides import overrides

import os
import copy
import time
import torch
import random
import numpy as np
import pandas as pd
from scripts.row_pop import RowPop

from torch.autograd import Variable
from allennlp.nn import util
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import FeedForward, TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy
# from allennlp.modules.token_embedders import PretrainedBertEmbedder
from table_embedder.models.lib.bert_token_embedder import PretrainedBertEmbedder
from allennlp.models.archival import load_archive

from table_embedder.models.embedder_util import TableUtil
from table_embedder.models.lib.stacked_self_attention import StackedSelfAttentionEncoder
# from lib.stacked_self_attention import StackedSelfAttentionEncoder

from table_embedder.models.cache_util import CacheUtil
from table_embedder.models.lib.masked_ff import MaskedFeedForward
# torch.set_default_tensor_type(torch.DoubleTensor)


from table_embedder.models.cache_util import CacheUtil

from torch import nn
from torch import tensor
import pdb

from torch_util.hypers_base import HypersBase


class TableEmbedderHypers(HypersBase):
    def __init__(self):
        super().__init__()
        self.num_max_row_pos = 31
        self.num_max_col_pos = 25
        self.n_classes = 300000
        self.n_seed_rows = 2
        # where we have clscol.npy  clsrow.npy
        self.saved_table_cls_dir = '/work/arnaik_umass_edu/tabbie/data'  # OR on CCC: '/dccstor/few-shot-rel/TableAugmentation/tabbie/data'
        self.model_file = '/work/arnaik_umass_edu/model_named.pt'
        
    def _post_init(self):
        super()._post_init()
        self.per_gpu_train_batch_size = 1
        if self.full_train_batch_size < self.world_size or self.full_train_batch_size % self.world_size != 0:
            raise ValueError(f'full_train_batch_size ({self.full_train_batch_size} '
                             f'must be a multiple of world_size ({self.world_size})')
        self.gradient_accumulation_steps = self.full_train_batch_size // self.world_size        


class TableEmbedder(torch.nn.Module):

    def __init__(self, hypers: TableEmbedderHypers) -> None:
        super().__init__()
        self.hypers = hypers
        
        self.row_pos_embedding = Embedding(num_embeddings=self.hypers.num_max_row_pos, embedding_dim=768)
        self.col_pos_embedding = Embedding(num_embeddings=self.hypers.num_max_col_pos, embedding_dim=768)
        
        ## NOTE: we again formulate this problem as multi-label classification, this time on top of the first column’s [CLSCOL] representation
        self.top_feedforward = MaskedFeedForward(input_dim = 768, num_layers = 1, hidden_dims= [self.hypers.n_classes])

        # # self.row_feedforward = row_feedforward # NOTE : Not used in forward pass
        # self.feedforward = feedforward # NOTE : Feedforward layer but not used in forward pass
        # self.compose_ff = compose_ff # NOTE : Feedforward layer but not used in forward pass
        
        self.bert_embedder = PretrainedBertEmbedder(pretrained_model='bert-base-uncased', top_layer_only=True)
        self.transformer_col1 = StackedSelfAttentionEncoder(768, 768, 768, 3072, 1, 12, False)
        self.transformer_col2 = StackedSelfAttentionEncoder(768, 768, 768, 3072, 1, 12, False)
        self.transformer_col3 = StackedSelfAttentionEncoder(768, 768, 768, 3072, 1, 12, False)
        self.transformer_col4 = StackedSelfAttentionEncoder(768, 768, 768, 3072, 1, 12, False)
        self.transformer_col5 = StackedSelfAttentionEncoder(768, 768, 768, 3072, 1, 12, False)
        self.transformer_col6 = StackedSelfAttentionEncoder(768, 768, 768, 3072, 1, 12, False)
        self.transformer_col7 = StackedSelfAttentionEncoder(768, 768, 768, 3072, 1, 12, False)
        self.transformer_col8 = StackedSelfAttentionEncoder(768, 768, 768, 3072, 1, 12, False)
        self.transformer_col9 = StackedSelfAttentionEncoder(768, 768, 768, 3072, 1, 12, False)
        self.transformer_col10 = StackedSelfAttentionEncoder(768, 768, 768, 3072, 1, 12, False)
        self.transformer_col11 = StackedSelfAttentionEncoder(768, 768, 768, 3072, 1, 12, False)
        self.transformer_col12 = StackedSelfAttentionEncoder(768, 768, 768, 3072, 1, 12, False)

        self.transformer_row1 = StackedSelfAttentionEncoder(768, 768, 768, 3072, 1, 12, False)
        self.transformer_row2 = StackedSelfAttentionEncoder(768, 768, 768, 3072, 1, 12, False)
        self.transformer_row3 = StackedSelfAttentionEncoder(768, 768, 768, 3072, 1, 12, False)
        self.transformer_row4 = StackedSelfAttentionEncoder(768, 768, 768, 3072, 1, 12, False)
        self.transformer_row5 = StackedSelfAttentionEncoder(768, 768, 768, 3072, 1, 12, False)
        self.transformer_row6 = StackedSelfAttentionEncoder(768, 768, 768, 3072, 1, 12, False)
        self.transformer_row7 = StackedSelfAttentionEncoder(768, 768, 768, 3072, 1, 12, False)
        self.transformer_row8 = StackedSelfAttentionEncoder(768, 768, 768, 3072, 1, 12, False)
        self.transformer_row9 = StackedSelfAttentionEncoder(768, 768, 768, 3072, 1, 12, False)
        self.transformer_row10 = StackedSelfAttentionEncoder(768, 768, 768, 3072, 1, 12, False)
        self.transformer_row11 = StackedSelfAttentionEncoder(768, 768, 768, 3072, 1, 12, False)
        self.transformer_row12 = StackedSelfAttentionEncoder(768, 768, 768, 3072, 1, 12, False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loss = torch.nn.BCEWithLogitsLoss()
#        self.loss_func = torch.nn.CrossEntropyLoss()

        self.cache_usage = None

        # TODO: try these as plain parameters, initialized from the saved clscol and clsrow embeddings
        self.cls_col = np.load(os.path.join(hypers.saved_table_cls_dir, 'clscol.npy'))
        self.cls_row = np.load(os.path.join(hypers.saved_table_cls_dir, 'clsrow.npy'))

        self.opt_level = 'O0'

        ## TODO : Updated the 'getenv'
        if self.cache_usage is not None:
            self.cache_util = CacheUtil(self.cache_usage, os.getenv("cell_db_path"))
        else:
            self.cache_util = None

        self.init_weight() # As feedforward layer is not present in the archieved model rather than using 'load_state_dict' still updating the model via a function
        
        # # TODO : Check what the load_label function does
        # self.label = RowPop.load_label(os.getenv('label_path'), key='index')

    def init_weight(self):
        model_parameters = dict(self.named_parameters())
        archived_parameters = torch.load(self.hypers.model_file) # Archieved weights of feedforward layers discarded.
        for name, weights in archived_parameters.items():
            if name in model_parameters:
                new_weights = weights.data
                model_parameters[name].data.copy_(new_weights)

    def get_tabemb(self, bert_header, bert_data, n_rows, n_cols, bs, table_mask, nrows, ncols):
        row_pos_ids = torch.arange(0, self.hypers.num_max_row_pos, device=self.device, dtype=torch.long)
        col_pos_ids = torch.arange(0, self.hypers.num_max_col_pos, device=self.device, dtype=torch.long)

        n_rows += 1  # row CLS
        n_cols += 1  # col CLS
        cls_col = torch.from_numpy(copy.deepcopy(self.cls_col)).to(device=self.device)
        cls_row = torch.from_numpy(copy.deepcopy(self.cls_row)).to(device=self.device)
        row_pos_embs = self.row_pos_embedding(row_pos_ids[:n_rows+1])
        col_pos_embs = self.col_pos_embedding(col_pos_ids[:n_cols])

        for i in range(1, 13):
            transformer_row = getattr(self, 'transformer_row{}'.format(str(i)))
            transformer_col = getattr(self, 'transformer_col{}'.format(str(i)))
            if i == 1:
                bert_data = TableUtil.add_cls_tokens(bert_header, bert_data, cls_row, cls_col, bs, n_rows, n_cols)
                bert_data += row_pos_embs.expand((bs, n_cols, n_rows + 1, 768)).permute(0, 2, 1, 3).expand_as(bert_data)
                bert_data += col_pos_embs.expand((bs, n_rows + 1, n_cols, 768)).expand_as(bert_data)
                # table_mask = TableUtil.add_cls_mask(table_mask, table_info, bs, n_rows, n_cols, self.device)
                table_mask = TableUtil.add_cls_mask(table_mask, bs, n_rows, n_cols, self.device, nrows, ncols)
                col_embs = TableUtil.get_col_embs(bert_data, bs, n_rows, n_cols, table_mask, transformer_col, self.opt_level)
                row_embs = TableUtil.get_row_embs(bert_data, bs, n_rows, n_cols, table_mask, transformer_row, self.opt_level)
            else:
                row_embs = TableUtil.get_row_embs(ave_embs, bs, n_rows, n_cols, table_mask, transformer_row, self.opt_level)
                col_embs = TableUtil.get_col_embs(ave_embs, bs, n_rows, n_cols, table_mask, transformer_col, self.opt_level)
            ave_embs = (row_embs + col_embs) / 2.0
        return row_embs, col_embs, n_rows, n_cols
    
    ## NOTE : Updating the function to not work for a batch process
    def pred_prob(self, cell_embs, labels, mask):
        out_prob = self.top_feedforward(cell_embs, mask)
        out_prob_1d = torch.cat([out_prob[0,:].expand(len(labels),out_prob.shape[1])], dim = 0)
        return out_prob_1d, out_prob

    @staticmethod
    def add_metadata(table_info, output_dict, pred_labels, pred_labels_name):
        data_dict = {'pred_labels': pred_labels, 'pred_labels_name': pred_labels_name}
        for one_info in table_info:
            for k, v in one_info.items():
                data_dict[k] = data_dict.get(k, [])
                data_dict[k].append(v)
        output_dict.update(data_dict)
        return output_dict

    def get_pred_labels(self, out_prob, labels, top_k=-1):
        pred_labels = []
        pred_labels_name = []
        for k, row_labels in enumerate(labels):
            n_pred = len(row_labels) if top_k == -1 else top_k
            pred_row_labels = out_prob[k][1:].argsort(dim=0, descending=True)[:n_pred].cpu().numpy()  # out_prob[0]: blank header
            pred_row_labels = [elem+1 for elem in pred_row_labels]  # add idx to 1 (for out_prob[0])
            pred_labels.append(pred_row_labels)
            pred_labels_name.append([self.label[elem] for elem in pred_row_labels])
        return pred_labels, pred_labels_name

    def sample_labels(self, labels_1d):
        """
        :return: mask : torch tensor with the true labels as 2, non-included labels as 0, false labels as 1
        :return: new_labels_1d: tensor of indices with shape: bs x len(labels_1d)
        :return: new_labels_1d: 0 or 1 tensor with shape: len(labels_1d) x self.hypers.n_classes        
        """
        # sampled label
        mask_bool = torch.cuda.FloatTensor(self.hypers.n_classes).uniform_() > 0.8
        mask = torch.tensor(mask_bool, dtype=torch.int)
        mask[labels_1d.long()] = 2

        # target label
        new_labels = mask[mask!=0]
        new_labels[new_labels==1] = 0
        new_labels[new_labels==2] = 1

        # old_new label map
        old_idx = (mask==2).nonzero()
        new_idx = (new_labels==1).nonzero()
        idx_map = {}
        for k, idx in enumerate(old_idx):
            idx_map[int(idx)] = int(new_idx[k])
        new_labels_1d = torch.autograd.Variable(labels_1d.clone())
        for k, idx in enumerate(labels_1d):
            new_labels_1d[k] = idx_map[int(idx)]
                   
        # New one-hot vector of labels
        new_labels_mat = torch.zeros(len(labels_1d),len(new_labels), dtype=torch.float).to(self.device)
        for k, idx in enumerate(new_labels_1d):
            new_labels_mat[k,idx] = 1.0
        return mask, new_labels_1d, new_labels_mat

    def validate_seed_rows(self, table_info):
        for one_info in table_info:
            if one_info['num_rows'] > self.n_seed_rows:
                raise ValueError('invalid num rows')

    def get_meta(self, table_info):
        nrows = [one_info['num_rows'] for one_info in table_info]
        ncols = [one_info['num_cols'] for one_info in table_info]
        tids = [one_info['table_id'] for one_info in table_info]
        return nrows, ncols, tids

    @overrides
    def forward(self,
                table_info: Dict[str, str],
                indexed_headers: Dict[str, torch.LongTensor],
                indexed_cells: Dict[str, torch.LongTensor],
                labels: Optional[torch.LongTensor]) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """

        :param table_info: see data/table_info.txt for example
        :param indexed_headers:  see data/indexed_headers.txt for example
        :param indexed_cells: see data/indexed_headers.txt for example
        :param labels: label idx mentioned based on word_vocab
        :return: loss and output probabilities (for each possible header)
        """
        
        # initialize
        self.bert_embedder.eval()
        # self.validate_seed_rows(table_info)
        bs, n_rows, n_cols = TableUtil.get_max_row_col(table_info)
        nrows, ncols, tids = self.get_meta(table_info)
        table_mask = TableUtil.get_table_mask(table_info, bs, n_rows, n_cols, self.device)

        # pred prob
        bert_header, bert_cell = TableUtil.get_bert_emb(indexed_headers, indexed_cells, table_info, bs, n_rows, n_cols
                                                        , self.cache_usage, self.bert_embedder, self.cache_util, self.device)
        row_embs, col_embs, n_rows_cls, n_cols_cls = self.get_tabemb(bert_header, bert_cell, n_rows, n_cols, bs, table_mask, nrows, ncols)
        ## NOTE: we again formulate this problem as multi-label classification, this time on top of the first column’s [CLSCOL] representation
        ## [bs, row having clscol, [CLSCOL] of 1st column, emb_dim]
        cell_embs = (col_embs[:, 0, 1, :] + row_embs[:, 0, 1, :]) / 2.0

        ## labels
        mask, sampled_labels_1d, sampled_labels = self.sample_labels(labels)
        out_prob, out_prob_1d = self.pred_prob(cell_embs, labels, mask)

        # Calculate Loss
        loss = self.loss(out_prob, sampled_labels)
        if self.training:
            if loss.isnan():
                pdb.set_trace()

        return loss, out_prob_1d, sampled_labels.sum(axis = 0)
    
if __name__ == "__main__":
    ## Sample Input
    table_info = [{'table_id': '1438042981460.12__CC-MAIN-20150728002301-00003-ip-10-236-191-2.ec2.internal__1387'
                   , 'num_rows': 2
                   , 'num_cols': 7
                   , 'header': ['size', 'us', 'type', 'china', 'gaa', 'eu', 'years']
                   , 'cell_labels': None
                   , 'col_labels': None
                   , 'table_labels': None
                   , 'table_data_raw': [['xs', '6/7', '9.125', '235', '23', '36-37', '4']
                                        , ['s', '7/8', '9.5', '245', '24', '38-39', '5']]
                   , 'table': [['size', 'us', 'type', 'china', 'gaa', 'eu', 'years']
                               , ['xs', '6/7', '9.125', '235', '23', '36-37', '4']
                               , ['s', '7/8', '9.5', '245', '24', '38-39', '5']]}]

    indexed_headers = {'bert': {'token_ids': tensor([[[  101,  2946,   102],
             [  101,  2149,   102],
             [  101,  2828,   102],
             [  101,  2859,   102],
             [  101, 19930,   102],
             [  101,  7327,   102],
             [  101,  2086,   102]]], device='cuda:0')
    , 'mask': tensor([[[True, True, True],
             [True, True, True],
             [True, True, True],
             [True, True, True],
             [True, True, True],
             [True, True, True],
             [True, True, True]]], device='cuda:0')
    , 'type_ids': tensor([[[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]], device='cuda:0')}}

    indexed_cells = {'bert': {'token_ids': tensor([[[[  101,  1060,  2015,   102,     0],
              [  101,  1020,  1013,  1021,   102],
              [  101,  1023,  1012,  8732,   102],
              [  101, 17825,   102,     0,     0],
              [  101,  2603,   102,     0,     0],
              [  101,  4029,  1011,  4261,   102],
              [  101,  1018,   102,     0,     0]],

             [[  101,  1055,   102,     0,     0],
              [  101,  1021,  1013,  1022,   102],
              [  101,  1023,  1012,  1019,   102],
              [  101, 21005,   102,     0,     0],
              [  101,  2484,   102,     0,     0],
              [  101,  4229,  1011,  4464,   102],
              [  101,  1019,   102,     0,     0]]]], device='cuda:0')
    , 'mask': tensor([[[[ True,  True,  True,  True, False],
              [ True,  True,  True,  True,  True],
              [ True,  True,  True,  True,  True],
              [ True,  True,  True, False, False],
              [ True,  True,  True, False, False],
              [ True,  True,  True,  True,  True],
              [ True,  True,  True, False, False]],

             [[ True,  True,  True, False, False],
              [ True,  True,  True,  True,  True],
              [ True,  True,  True,  True,  True],
              [ True,  True,  True, False, False],
              [ True,  True,  True, False, False],
              [ True,  True,  True,  True,  True],
              [ True,  True,  True, False, False]]]], device='cuda:0')
    , 'type_ids': tensor([[[[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]],

             [[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]]]], device='cuda:0')}}

    hypers = TableEmbedderHypers()
    model = TableEmbedder(hypers).to('cuda:0')
    loss, prob = model(table_info, indexed_headers, indexed_cells, None)
    print(prob.shape)