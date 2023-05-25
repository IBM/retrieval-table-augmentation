from table_embedder.models.finetune_entitables_cp import TableEmbedderHypers, TableEmbedder
from torch_util.transformer_optimize import TransformerOptimize, LossHistory
from torch_util.line_corpus import jsonl_lines, block_shuffle
import json
from table_embedder.table_info_tokenize import tokenize_table_info
from pytorch_pretrained_bert import BertTokenizer
from typing import Mapping, List, Tuple
import torch
import os
import pdb
from tqdm import tqdm
import numpy as np
import random

import logging
import sys
from datetime import datetime
from scripts.util import TqdmToLogger

dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M")
logging.basicConfig(filename='./log/'+dt_string+'.log'
                    , filemode='w'
                    , format="[%(asctime)s \t %(message)s]"
                    , datefmt='%H:%M:%S'
                    , level=logging.INFO)
# handler = logging.StreamHandler()
# handler.setLevel(logging.INFO)
# handler.addFilter(HostnameFilter())
# format = logging.Formatter('[%(asctime)s \t %(message)s]',
#                            datefmt='%m/%d/%Y %H:%M:%S')
# handler.setFormatter(format)
# logging.getLogger('').addHandler(handler)
logger = logging.getLogger(__name__)

def create_vocab(filename: str) -> Mapping[str, int]:
    vocab = {}
    i = 0
    for line in jsonl_lines(filename):
        for header in json.loads(line)['header']:
            if header not in vocab:
                vocab[header] = i
                i+=1
    return vocab

def to_table_info(entitable, n_seed_cols: int, num_max_row_pos: int) -> Tuple[List[dict], List[str]]:
    table_info = {}
    table_info['table_id'], table_info['num_rows'], table_info['num_cols'] = entitable['table_id'], min(num_max_row_pos-2,len(entitable['rows'])), min(n_seed_cols, len(entitable['header']))
    table_info['header'] = entitable['header'][0:table_info['num_cols']]
    table_info['cell_lables'], table_info['col_lables'], table_info['table_lables'] = None, None, None
    table_info['table_data_raw'] = []
    table_info['table'] = []
    table_info['table'].append(table_info['header'])
    other_headers = entitable['header'][table_info['num_cols']:]
    n_rows = 0
    ## NOTE: Truncating the number of rows in the table based on the parameter 'num_max_row_pos'
    for row in entitable['rows']:
        if n_rows < num_max_row_pos-2:
            table_info['table_data_raw'].append(row[0:table_info['num_cols']])
            table_info['table'].append(row[0:table_info['num_cols']])
            n_rows+=1
        else:
            break
    assert len(table_info['table_data_raw']) <= num_max_row_pos
    assert len(table_info['header']) <= n_seed_cols
    return [table_info], other_headers

def get_labels(other_headers: List[str], header_vocab: Mapping[str, int]) -> torch.FloatTensor:
    labels = torch.zeros(len(header_vocab), dtype=torch.float)
    for oh in other_headers:
        if oh in header_vocab:
            labels[header_vocab[oh]] = 1.0
    return labels.unsqueeze(0)

def evaluate(eval_type, hypers, tokenizer, header_vocab, model):
    if eval_type == 'test':
        filename = hypers.test_tables
    elif eval_type == 'train':
        filename = hypers.train_tables
    else:
        raise ValueError("Provide appropriate evaluation type")
    
    model.eval()
    instances = sum(1 for _ in jsonl_lines(filename))

    print("Number of instances for evaluation :", str(instances))
    reciprocal_rank = 0.0
    with tqdm(total=instances, desc='[EVALUATION]', leave=False) as pbar:
        for line_idx, line in enumerate(jsonl_lines(filename)):
            jobj_eval = json.loads(line)
            table_info, other_headers = to_table_info(jobj_eval, hypers.n_seed_cols, hypers.num_max_row_pos)
            indexed_headers, indexed_cells = tokenize_table_info(table_info, tokenizer,
                                                                 max_length=hypers.max_cell_length,
                                                                 device=hypers.device)
            if indexed_headers['bert']['token_ids'].shape[2] == 0:
                indexed_headers['bert']['token_ids'] = torch.zeros(indexed_headers['bert']['token_ids'].shape[0],indexed_headers['bert']['token_ids'].shape[1],1, dtype = torch.int64,device = hypers.device)
            labels = get_labels(other_headers, header_vocab).to(hypers.device)
            _, probs = model(table_info, indexed_headers, indexed_cells, None)
            rank = eval(probs[0,:].cpu().detach().numpy(), labels[0,:].cpu().detach().numpy())
            reciprocal_rank += 1.0 / rank if rank > 0 else 0.0
            pbar.update(1)
        mrr = reciprocal_rank / instances
        return mrr

def eval(probs,labels):
    score_correct = list(zip(probs, labels))
    score_correct.sort(key=lambda x: x[0], reverse=True)
    for zrank, sc in enumerate(score_correct):
        if sc[1]:
            return zrank + 1
    return -1

class ColumnPopulationOptions(TableEmbedderHypers):
    def __init__(self):
        super().__init__()
        self.train_tables = ''
        self.test_tables = ''
        self.test_batch_size = 1
        self.max_cell_length = 128
#        self.__required_args__ = ['train_tables', 'header_vocab', 'output_dir']
        self.__required_args__ = ['train_tables','output_dir','n_seed_cols']

def save(hypers: ColumnPopulationOptions, model):
    if hypers.global_rank == 0:
        os.makedirs(hypers.output_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(hypers, os.path.join(hypers.output_dir, "training_args.bin"))
        torch.save(model_to_save.state_dict(), os.path.join(hypers.output_dir, "model.bin"))


if __name__ == "__main__":
    hypers = ColumnPopulationOptions()
    hypers.fill_from_args()
    # NOTE: don't copy this for row population - need to filter out the train tables to those that have entities
    train_instances = sum(1 for _ in jsonl_lines(hypers.train_tables))
    print("Number of Training Instances : ", str(train_instances))
    logger.info("Number of Training Instances {}".format(train_instances))
    
    if hypers.test_tables is not '':
        test_instances = sum(1 for _ in jsonl_lines(hypers.test_tables))
        logger.info("Number of Test Instances {}".format(test_instances))
        print("Number of Test Instances : ", str(test_instances))
    header_vocab = create_vocab(hypers.train_tables)
    hypers.n_classes = len(header_vocab)
    logger.info("Number of Headers {}".format(hypers.n_classes))
    print("Number of Headers : ", str(hypers.n_classes))
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TableEmbedder(hypers).to(hypers.device)

    optimizer = TransformerOptimize(hypers, train_instances*hypers.num_train_epochs, model)
    rand = random.Random(hypers.seed)
    
    mrr = evaluate('test', hypers, tokenizer, header_vocab, model)
    print('MRR using initialized model: {}'.format(mrr))
    logger.info('MRR using initialized model: {}'.format(mrr))
    
    while optimizer.should_continue():
        with tqdm(total=train_instances, desc='[TRAIN]', leave=True) as pbar:
            for line_ndx, line in enumerate(block_shuffle(jsonl_lines(hypers.train_tables), rand=rand)):
                if not optimizer.should_continue():
                    break
                if line_ndx % hypers.world_size != hypers.global_rank:
                    continue  # in distributed training, skip over instances that are not ours
                jobj = json.loads(line)
                table_info, other_headers = to_table_info(jobj, hypers.n_seed_cols, hypers.num_max_row_pos)
                indexed_headers, indexed_cells = tokenize_table_info(table_info, tokenizer,
                                                                     max_length=hypers.max_cell_length,
                                                                     device=hypers.device)
                ## Creating Padded Headers
                if indexed_headers['bert']['token_ids'].shape[2] == 0:
                    indexed_headers['bert']['token_ids'] = torch.zeros(indexed_headers['bert']['token_ids'].shape[0]
                                                                       ,indexed_headers['bert']['token_ids'].shape[1],1
                                                                       , dtype = torch.int64
                                                                       ,device = hypers.device)
                    
                ## Skip training examples where no tokens are present. This is happening because train data is not unicode normalized.
                if indexed_cells['bert']['token_ids'].shape[-1] == 0:
                    pbar.update(1)
                    logger.info("Skipping training table : {}".format(str(table_info[0]['table_id'])))
                    continue
                labels = get_labels(other_headers, header_vocab).to(hypers.device)
                loss, probs = optimizer.model(table_info, indexed_headers, indexed_cells, labels)
                if (line_ndx+1) % 50000 == 0:
                    mrr = evaluate('test', hypers, tokenizer, header_vocab, model)
                    print('MRR using {} model: {}'.format(line_ndx,str(mrr)))
                    logger.info('MRR using {} model: {}'.format(line_ndx,str(mrr)))
                if (line_ndx) % 200000 == 0:
                    logger.info("Saving model with MRR {} at step {}.".format(mrr,line_ndx))
                    save(hypers, optimizer.model)
                optimizer.step_loss(loss)
                pbar.update(1)