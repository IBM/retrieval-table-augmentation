from typing import Dict, Optional, Tuple
from overrides import overrides

import os
import copy
import torch
import numpy as np
from pathlib import Path
from scripts.to_npy import ToNpy

from allennlp.nn import util
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import FeedForward, TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy
from allennlp.nn.activations import Activation
from allennlp.predictors.predictor import Predictor

from table_embedder.models.lib.bert_token_embedder import PretrainedBertEmbedder

from table_embedder.models.embedder_util import TableUtil
# from embedder_util import TableUtil
# from embedder_util import PredUtil
from table_embedder.models.lib.stacked_self_attention import StackedSelfAttentionEncoder
# from lib.stacked_self_attention import StackedSelfAttentionEncoder
from allennlp.models.archival import load_archive

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
        self.n_classes = 127656  # FIXME: we'll update this, since I can't reproduce
        self.n_seed_cols = 2
        # where we have clscol.npy  clsrow.npy
        self.saved_table_cls_dir = '/work/arnaik_umass_edu/tabbie/data'  # OR on CCC: '/dccstor/few-shot-rel/TableAugmentation/tabbie/data'
        self.model_file = '/work/arnaik_umass_edu/model_named.pt'  # OR on CCC: '/dccstor/few-shot-rel/TableAugmentation/tabbie/model/model_named.pt'

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

        ## As per the paper n_seed_cols = 1/2/3
        self.row_pos_embedding = Embedding(num_embeddings=self.hypers.num_max_row_pos, embedding_dim=768)
        self.col_pos_embedding = Embedding(num_embeddings=self.hypers.num_max_col_pos, embedding_dim=768)

        # NOTE: this is the only parameter not initialized from the pretrained TABBIE model
        # NOTE: Activation used in linear i.e. basically as per allennlp module it is ignored - https://github.com/allenai/allennlp/blob/master/allennlp/nn/activations.py#L78
        # TODO: are you sure the feedforward depends on the number of seeds? I think this should be 768*3 - The layer before the feedforward layer concatenates all COLCLS tokens 
        ## using function "get_cat_cls" wherein the for-loop uses n_seed_cols to get the concatenated vector.
        
        self.top_feedforward = nn.Linear(768*self.hypers.n_seed_cols, self.hypers.n_classes)
        self.Dropout = nn.Dropout(p = 0.0) # At the moment Dropout not used as only one-layer network
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

        ## TODO : Provide cache_usage as parameter, currently set to None
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
        # NOTE: there is no need for a copy, but it doesn't really matter
        cls_col = torch.from_numpy(copy.deepcopy(self.cls_col)).to(device=self.device)
        cls_row = torch.from_numpy(copy.deepcopy(self.cls_row)).to(device=self.device)
        row_pos_embs = self.row_pos_embedding(row_pos_ids[:n_rows + 1])
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
                col_embs = TableUtil.get_col_embs(bert_data, bs, n_rows, n_cols, table_mask, transformer_col,
                                                  self.opt_level)
                row_embs = TableUtil.get_row_embs(bert_data, bs, n_rows, n_cols, table_mask, transformer_row,
                                                  self.opt_level)
            else:
                row_embs = TableUtil.get_row_embs(ave_embs, bs, n_rows, n_cols, table_mask, transformer_row,
                                                  self.opt_level)
                col_embs = TableUtil.get_col_embs(ave_embs, bs, n_rows, n_cols, table_mask, transformer_col,
                                                  self.opt_level)
            ave_embs = (row_embs + col_embs) / 2.0
        return row_embs, col_embs, n_rows, n_cols

    @staticmethod
    def get_cat_cls(row_embs, col_embs, n_seed_cols, device):
        # NOTE: Not just colcls concatenated but their average with row_embs concatenated
        ave_embs = (row_embs + col_embs) / 2.0
        ave_embs = ave_embs[:, 0, 1:, :] # [bs, row having clscol, first_row:all_rows, emb_dim]
        bs = ave_embs.shape[0]

        cls_embs = ave_embs[:, 0, :]
        for i in range(1, n_seed_cols):
            if ave_embs.shape[1] <= i: # This condition is never met as num clscol embeddings == num_seed_cols
                zeros = torch.zeros((bs, 768), device=device)
                cls_embs = torch.cat([cls_embs, zeros], dim=1)
            else:
                cls_embs = torch.cat([cls_embs, ave_embs[:, i, :]], dim=1)
        return cls_embs

    @staticmethod
    def mask_cls_embs(cls_embs, table_info):
        # NOTE: Function to mask cls embedding based on number of columns / number of seed columns present in table
        for k, one_info in enumerate(table_info):
            if one_info['num_cols'] == 1:
                cls_embs[k, 768:] = 0
            elif one_info['num_cols'] == 2:
                cls_embs[k, (768 * 2):] = 0
        return cls_embs

    def validate_seed_cols(self, table_info):
        for one_info in table_info:
            if one_info['num_cols'] > self.hypers.n_seed_cols:
                raise ValueError('invalid num cols')

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
                labels: Optional[torch.FloatTensor]) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """

        :param table_info: see data/table_info.txt for example
        :param indexed_headers:  see data/indexed_headers.txt for example
        :param indexed_cells: see data/indexed_headers.txt for example
        :param labels: 0 or 1 tensor with shape: bs x self.hypers.n_classes
        :return: loss and output probabilities (for each possible header)
        """
        ## Initialize
        self.bert_embedder.eval()
        self.validate_seed_cols(table_info)
        bs, n_rows, n_cols = TableUtil.get_max_row_col(table_info)
        nrows, ncols, tids = self.get_meta(table_info)
        table_mask = TableUtil.get_table_mask(table_info, bs, n_rows, n_cols, self.device)

        ## Prediction Probability
        bert_header, bert_cell = TableUtil.get_bert_emb(indexed_headers, indexed_cells, table_info, bs, n_rows, n_cols,
                                                        self.cache_usage, self.bert_embedder, self.cache_util,
                                                        self.device)

        row_embs, col_embs, n_rows_cls, n_cols_cls = self.get_tabemb(bert_header, bert_cell, n_rows, n_cols, bs,
                                                                     table_mask, nrows, ncols)

        # To fine-tune TABBIE on this task, we first concatenate the column [CLSCOL]
        # embeddings of the seed table into a single vector and pass it through a single linear
        # and softmax layer, training with a multi-label classification objective

        cls_embs = self.get_cat_cls(row_embs, col_embs, self.hypers.n_seed_cols, self.device) # Concatenate clscol embeddings for seed columns
        cls_embs = self.mask_cls_embs(cls_embs, table_info)
        out_prob = self.top_feedforward(cls_embs)
        
        ## Compute Loss
        if labels is not None:
            loss = self.loss(out_prob, labels)
        else:
            loss = None

        return loss, out_prob


if __name__ == "__main__":
    ## Sample Input
    table_info = [{'table_id': '1438042981460.12__CC-MAIN-20150728002301-00003-ip-10-236-191-2.ec2.internal__1387'
                      , 'num_rows': 3
                      , 'num_cols': 2
                      , 'header': ['size', 'japan (cm)']
                      , 'cell_labels': None
                      , 'col_labels': None
                      , 'table_labels': None
                      , 'table_data_raw': [['6/7', '23']
            , ['s', '24']
            , ['m', '25']]
                      , 'table': [['size', 'japan (cm)']
            , ['6/7', '23']
            , ['s', '24']
            , ['m', '25']]},{'table_id': '1438042981460.12__CC-MAIN-20150728002301-00003-ip-10-236-191-2.ec2.internal__1387'
                      , 'num_rows': 3
                      , 'num_cols': 2
                      , 'header': ['size', 'japan (cm)']
                      , 'cell_labels': None
                      , 'col_labels': None
                      , 'table_labels': None
                      , 'table_data_raw': [['6/7', '23']
            , ['s', '24']
            , ['m', '25']]
                      , 'table': [['size', 'japan (cm)']
            , ['6/7', '23']
            , ['s', '24']
            , ['m', '25']]},{'table_id': '1438042981460.12__CC-MAIN-20150728002301-00003-ip-10-236-191-2.ec2.internal__1387'
                      , 'num_rows': 3
                      , 'num_cols': 2
                      , 'header': ['size', 'japan (cm)']
                      , 'cell_labels': None
                      , 'col_labels': None
                      , 'table_labels': None
                      , 'table_data_raw': [['6/7', '23']
            , ['s', '24']
            , ['m', '25']]
                      , 'table': [['size', 'japan (cm)']
            , ['6/7', '23']
            , ['s', '24']
            , ['m', '25']]}]

    indexed_headers = {'bert': {'token_ids': tensor([[[101, 2946, 102, 0, 0, 0],
                                                      [101, 2900, 1006, 4642, 1007, 102]]
                                                     ,[[101, 2946, 102, 0, 0, 0],
                                                      [101, 2900, 1006, 4642, 1007, 102]]
                                                     ,[[101, 2946, 102, 0, 0, 0],
                                                      [101, 2900, 1006, 4642, 1007, 102]]], device='cuda:0')
        , 'mask': tensor([[[True, True, True, False, False, False],
                           [True, True, True, True, True, True]],
                         [[True, True, True, False, False, False],
                           [True, True, True, True, True, True]],
                         [[True, True, True, False, False, False],
                           [True, True, True, True, True, True]]], device='cuda:0')
        , 'type_ids': tensor([[[0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0]],
                             [[0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0]],
                             [[0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0]]], device='cuda:0')}}

    indexed_cells = {'bert': {'token_ids': tensor([[[[101, 1020, 1013, 1021, 102],[101, 2603, 102, 0, 0]],

                                                    [[101, 1055, 102, 0, 0],
                                                     [101, 2484, 102, 0, 0]],

                                                    [[101, 1049, 102, 0, 0],
                                                     [101, 2423, 102, 0, 0]]],
                                                  [[[101, 1020, 1013, 1021, 102],[101, 2603, 102, 0, 0]],

                                                    [[101, 1055, 102, 0, 0],
                                                     [101, 2484, 102, 0, 0]],

                                                    [[101, 1049, 102, 0, 0],
                                                     [101, 2423, 102, 0, 0]]],
                                                  [[[101, 1020, 1013, 1021, 102],[101, 2603, 102, 0, 0]],

                                                    [[101, 1055, 102, 0, 0],
                                                     [101, 2484, 102, 0, 0]],

                                                    [[101, 1049, 102, 0, 0],
                                                     [101, 2423, 102, 0, 0]]]], device='cuda:0')
        , 'mask': tensor([[[[True, True, True, True, True],
                            [True, True, True, False, False]],

                           [[True, True, True, False, False],
                            [True, True, True, False, False]],

                           [[True, True, True, False, False],
                            [True, True, True, False, False]]],
                         [[[True, True, True, True, True],
                            [True, True, True, False, False]],

                           [[True, True, True, False, False],
                            [True, True, True, False, False]],

                           [[True, True, True, False, False],
                            [True, True, True, False, False]]],
                         [[[True, True, True, True, True],
                            [True, True, True, False, False]],

                           [[True, True, True, False, False],
                            [True, True, True, False, False]],

                           [[True, True, True, False, False],
                            [True, True, True, False, False]]]], device='cuda:0')
        , 'type_ids': tensor([[[[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]]],
                             [[[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]]],
                             [[[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]]]], device='cuda:0')}}

    hypers = TableEmbedderHypers()
    model = TableEmbedder(hypers).to('cuda:0')
    loss, prob = model(table_info, indexed_headers, indexed_cells, None)
    print(prob.shape)