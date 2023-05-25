from table_embedder.models.finetune_entitables_rp import TableEmbedderHypers, TableEmbedder
from torch_util.transformer_optimize import TransformerOptimize, LossHistory
from torch_util.line_corpus import jsonl_lines, block_shuffle
import json
from collections import defaultdict
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
logger = logging.getLogger(__name__)

## TODO : Clean train.jsonl.gz as per previous cleaning steps on train.jsonl
def create_vocab(filename: str, num_instances: int, min_count: int) -> Mapping[str, int]:
    vocab = {}
    trunc_vocab = []
    filename = './data/queries/train.jsonl.gz'
    for line_idx, line in enumerate(jsonl_lines(filename)):
        cell_list = np.array(json.loads(line)['rows'])[np.array(json.loads(line)['candidate_cells']).astype(bool)].astype(list)
        for cell in cell_list:
            if cell in vocab and vocab[cell] >= min_count:
                trunc_vocab.append(cell)
            elif cell in vocab:
                vocab[cell] += 1.0
            elif cell not in vocab:
                vocab[cell] = 1.0
    print('Dict length:',str(len(set(trunc_vocab))))
    return dict(zip(set(trunc_vocab), list(range(0,len(set(trunc_vocab)), 1))))

def to_table_info(entitable, n_seed_rows: int, num_max_col_pos: int) -> Tuple[List[dict], List[str]]:
    table_info = {}
    table_info['table_id'], table_info['num_rows'], table_info['num_cols'] = entitable['table_id'], min(n_seed_rows,len(entitable['rows'])), min(num_max_col_pos-1, len(entitable['header']))
    table_info['header'] = entitable['header'][0:table_info['num_cols']]
    table_info['cell_lables'], table_info['col_lables'], table_info['table_lables'] = None, None, None  # mrglass 'lables'? I guess these aren't needed anyway
    table_info['table_data_raw'] = []
    table_info['table'] = []
    table_info['table'].append(table_info['header'])
    rows = entitable['rows']
    candidates = entitable['candidate_cells']
    answers = [row[0] for row, candidates_in_row in zip(rows[n_seed_rows:], candidates[n_seed_rows:]) if candidates_in_row[0] == 1]
    # other_row = [row[0] for row in entitable['rows'][table_info['num_rows']:]]  # mrglass: the answers are the other entities
    n_rows = 0

    ## NOTE: Truncating the number of rows in the table based on the parameter 'n_seed_rows'
    for row in entitable['rows']:
        if n_rows < n_seed_rows:
            table_info['table_data_raw'].append(row[0:table_info['num_cols']])
            table_info['table'].append(row[0:table_info['num_cols']])
            n_rows+=1
        else:
            break
    assert len(table_info['table_data_raw']) <= n_seed_rows
    assert len(table_info['header']) <= num_max_col_pos
    assert len(table_info['table_data_raw'][0]) <= num_max_col_pos
    assert len(table_info['table'][0]) <= num_max_col_pos
    return [table_info], answers

def get_labels(other_row: List[str], word_vocab: Mapping[str, int]) -> torch.FloatTensor:
    labels_idx = torch.zeros(len(other_row), dtype=torch.long)
    for idx, oh in enumerate(other_row):
        if oh in word_vocab:
            labels_idx[idx] = word_vocab[oh]
        else:
            labels_idx[idx] = -1000
    return labels_idx[labels_idx!=-1000] # Remove lables which have index -1000 as they are not present in vocabulary

## TODO : Fix Me
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
            table_info, other_headers = to_table_info(jobj_eval, hypers.n_seed_rows, hypers.num_max_col_pos)
            indexed_headers, indexed_cells = tokenize_table_info(table_info, tokenizer,
                                                                 max_length=hypers.max_cell_length,
                                                                 device=hypers.device)
            if indexed_headers['bert']['token_ids'].shape[2] == 0:
                indexed_headers['bert']['token_ids'] = torch.zeros(indexed_headers['bert']['token_ids'].shape[0],indexed_headers['bert']['token_ids'].shape[1],1, dtype = torch.int64,device = hypers.device)
            labels_idx = get_labels(other_headers, header_vocab).to(hypers.device)
            _, probs, sampled_labels = model(table_info, indexed_headers, indexed_cells, labels_idx)
            rank = eval(probs[0,:].cpu().detach().numpy(), sampled_labels.cpu().detach().numpy())
            reciprocal_rank += 1.0 / rank if rank > 0 else 0.0
            pbar.update(1)
        mrr = reciprocal_rank / instances
        return mrr
    
## TODO : Fix Me
def eval(probs,labels):
    score_correct = list(zip(probs, labels))
    score_correct.sort(key=lambda x: x[0], reverse=True)
    for zrank, sc in enumerate(score_correct):
        if sc[1]:
            return zrank + 1
    return -1

class RowPopulationOptions(TableEmbedderHypers):
    def __init__(self):
        super().__init__()
        self.train_tables = ''
        self.test_tables = ''
        self.test_batch_size = 1
        self.max_cell_length = 128
        self.min_fraction_entities_for_train = 0.3
        self.__required_args__ = ['train_tables','output_dir','n_seed_rows']

def save(hypers: RowPopulationOptions, model):
    if hypers.global_rank == 0:
        os.makedirs(hypers.output_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(hypers, os.path.join(hypers.output_dir, "training_args.bin"))
        torch.save(model_to_save.state_dict(), os.path.join(hypers.output_dir, "model.bin"))


if __name__ == "__main__":
    hypers = RowPopulationOptions()
    hypers.fill_from_args()

    # NOTE: don't copy this for row population - need to filter out the train tables to those that have entities
    train_instances = sum(1 for _ in jsonl_lines(hypers.train_tables))
    print("Number of Training Instances : ", str(train_instances))
    logger.info("Number of Training Instances {}".format(train_instances))
    
    if hypers.test_tables is not '':
        test_instances = sum(1 for _ in jsonl_lines(hypers.test_tables))
        logger.info("Number of Test Instances {}".format(test_instances))
        print("Number of Test Instances : ", str(test_instances))
    
    word_vocab = create_vocab(hypers.train_tables+'.gz', train_instances, 7.0) # Only for row population a .jsonl.gz file sent; Vocab size of 319,634
    hypers.n_classes = len(word_vocab)
    logger.info("Number of Words {}".format(hypers.n_classes))
    print("Number of Words : ", str(hypers.n_classes))
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TableEmbedder(hypers).to(hypers.device)

    optimizer = TransformerOptimize(hypers, train_instances*hypers.num_train_epochs, model)
    rand = random.Random(hypers.seed)
    
    mrr = evaluate('test', hypers, tokenizer, word_vocab, model)
    print('MRR using initialized model: {}'.format(mrr))
    logger.info('MRR using initialized model: {}'.format(mrr))
    
    model.train()
    while optimizer.should_continue():
        with tqdm(total=train_instances, desc='[TRAIN]', leave=True) as pbar:
            for line_ndx, line in enumerate(block_shuffle(jsonl_lines(hypers.train_tables), rand=rand)):
                if not optimizer.should_continue():
                    break
                if line_ndx % hypers.world_size != hypers.global_rank:
                    continue  # in distributed training, skip over instances that are not ours
                jobj = json.loads(line)
                table_info, other_row_words = to_table_info(jobj, hypers.n_seed_rows, hypers.num_max_col_pos)
                indexed_headers, indexed_cells = tokenize_table_info(table_info, tokenizer,max_length=hypers.max_cell_length,device=hypers.device)
                
                ## Creating Padded Headers
                if indexed_headers['bert']['token_ids'].shape[-1] == 0:
                    indexed_headers['bert']['token_ids'] = torch.zeros(indexed_headers['bert']['token_ids'].shape[0]
                                                                       ,indexed_headers['bert']['token_ids'].shape[1],1
                                                                       , dtype = torch.int64
                                                                       ,device = hypers.device)
                    
                ## Skip training examples where no tokens are present. This is happening because train data is not unicode normalized.
                if indexed_cells['bert']['token_ids'].shape[-1] == 0:
                    pbar.update(1)
                    logger.info("Skipping training table : {}".format(str(table_info[0]['table_id'])))
                    continue
                label_idx = get_labels(other_row_words, word_vocab).to(hypers.device)

                ## Labels could be empty as we are down sampling the labels
                if len(label_idx) >= hypers.min_fraction_entities_for_train * (len(jobj['rows']) - hypers.n_seed_rows) and len(label_idx) > 0:
                    pbar.update(1)
                    continue
                    
                loss, _, _ = optimizer.model(table_info, indexed_headers, indexed_cells, label_idx)
                if (line_ndx+1) % 10000 == 0:
                    mrr = evaluate('test', hypers, tokenizer, word_vocab, model)
                    print('MRR using {} model: {}'.format(line_ndx,str(mrr)))
                    logger.info('MRR using {} model: {}'.format(line_ndx,str(mrr)))
                if (line_ndx) % 200000 == 0:
                    logger.info("Saving model with MRR {} at step {}.".format(mrr,line_ndx))
                    save(hypers, optimizer.model)
                optimizer.step_loss(loss)
                pbar.update(1)