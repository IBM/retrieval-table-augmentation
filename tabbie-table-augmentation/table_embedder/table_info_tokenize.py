from pytorch_pretrained_bert import BertTokenizer
from typing import List, Mapping
import torch
import pdb


def batch_tokenize(tokenizer: BertTokenizer, texts: List[str], *, max_length=128) -> Mapping[str, torch.Tensor]:
    token_ids_list = []
    for text in texts:
        token_ids_list.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))[:max_length])
    length = max([len(toks) for toks in token_ids_list])
    token_ids = torch.zeros((len(texts), length), dtype=torch.long)
    mask = torch.zeros((len(texts), length), dtype=torch.bool)
    type_ids = torch.zeros((len(texts), length), dtype=torch.long)
    for ndx, tokens in enumerate(token_ids_list):
        token_ids[ndx, :len(tokens)] = torch.tensor(tokens)
        mask[ndx, :len(tokens)] = True
    return {'token_ids': token_ids, 'mask': mask, 'type_ids': type_ids}


def tokenize_table_info(table_info: List[dict], tokenizer: BertTokenizer, *, device=None, max_length=128):
    """
    from the table_info text datastructure passed to table_embedder.models.finetune_*.py
    :param table_info:
    :param tokenizer:
    :param device:
    :param max_length:
    :return: we produce the indexed_headers and indexed_cells
    """
    assert len(table_info) == 1
    header = table_info[0]['header']
    rows = table_info[0]['table_data_raw']
    n_rows = len(rows)
    n_cols = len(header)
    indexed_headers = {name: t.reshape(1, n_cols, -1).to(device) for name, t in
                       batch_tokenize(tokenizer, header, max_length=max_length).items()}
    row_cells = [cell for row in rows for cell in row]
    assert len(row_cells) == n_rows*n_cols, "Table ID : {}".format(table_info[0]['table_id'])
    indexed_cells = {name: t.reshape(1, n_rows, n_cols, -1).to(device) for name, t in
                     batch_tokenize(tokenizer, row_cells, max_length=max_length).items()}
    return {'bert': indexed_headers}, {'bert': indexed_cells}


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print(batch_tokenize(tokenizer, ['size', 'us', 'inches', 'china', 'japan (cm)', 'eu', 'uk']))

    """
    {'token_ids': tensor([[2946,    0,    0,    0],
        [2149,    0,    0,    0],
        [5282,    0,    0,    0],
        [2859,    0,    0,    0],
        [2900, 1006, 4642, 1007],
        [7327,    0,    0,    0],
        [2866,    0,    0,    0]]), 'mask': tensor([[ True, False, False, False],
        [ True, False, False, False],
        [ True, False, False, False],
        [ True, False, False, False],
        [ True,  True,  True,  True],
        [ True, False, False, False],
        [ True, False, False, False]]), 'type_ids': tensor([[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]])}
    """


    table_info = [{'table_id': '1438042981460.12__CC-MAIN-20150728002301-00003-ip-10-236-191-2.ec2.internal__1387',
                   'num_rows': 4, 'num_cols': 7,
                   'header': ['size', 'us', 'inches', 'china', 'japan (cm)', 'eu', 'uk'],
                   'table_data_raw': [['6/7', 'xs', '9.125', '235', '23', '36-37', '4'],
                                     ['s', '7/8', '40-41', '245', '24', '38-39', '5'],
                                     ['m', '8/9', '9.875', '255', '25', '39-40', '6'],
                                     ['l', '9/10', '10.375', '265', '26', '9.5', '7']]
                                    ,
                   'table': [['size', 'us', 'inches', 'china', 'japan (cm)', 'eu', 'uk'],
                            ['6/7', 'xs', '9.125', '235', '23', '36-37', '4'],
                            ['s', '7/8', '40-41', '245', '24', '38-39', '5'],
                            ['m', '8/9', '9.875', '255', '25', '39-40', '6'],
                            ['l', '9/10', '10.375', '265', '26', '9.5', '7']]}]

    print(tokenize_table_info(table_info, tokenizer))

    """
    ({'bert': {'token_ids': tensor([[[2946,    0,    0,    0],
         [2149,    0,    0,    0],
         [5282,    0,    0,    0],
         [2859,    0,    0,    0],
         [2900, 1006, 4642, 1007],
         [7327,    0,    0,    0],
         [2866,    0,    0,    0]]]), 'mask': tensor([[[ True, False, False, False],
         [ True, False, False, False],
         [ True, False, False, False],
         [ True, False, False, False],
         [ True,  True,  True,  True],
         [ True, False, False, False],
         [ True, False, False, False]]]), 'type_ids': tensor([[[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]])}}, {'bert': {'token_ids': tensor([[[[ 1020,  1013,  1021],
          [ 1060,  2015,     0],
          [ 1023,  1012,  8732],
          [17825,     0,     0],
          [ 2603,     0,     0],
          [ 4029,  1011,  4261],
          [ 1018,     0,     0]],

         [[ 1055,     0,     0],
          [ 1021,  1013,  1022],
          [ 2871,  1011,  4601],
          [21005,     0,     0],
          [ 2484,     0,     0],
          [ 4229,  1011,  4464],
          [ 1019,     0,     0]],

         [[ 1049,     0,     0],
          [ 1022,  1013,  1023],
          [ 1023,  1012, 27658],
          [20637,     0,     0],
          [ 2423,     0,     0],
          [ 4464,  1011,  2871],
          [ 1020,     0,     0]],

         [[ 1048,     0,     0],
          [ 1023,  1013,  2184],
          [ 2184,  1012, 18034],
          [20549,     0,     0],
          [ 2656,     0,     0],
          [ 1023,  1012,  1019],
          [ 1021,     0,     0]]]]), 'mask': tensor([[[[ True,  True,  True],
          [ True,  True, False],
          [ True,  True,  True],
          [ True, False, False],
          [ True, False, False],
          [ True,  True,  True],
          [ True, False, False]],

         [[ True, False, False],
          [ True,  True,  True],
          [ True,  True,  True],
          [ True, False, False],
          [ True, False, False],
          [ True,  True,  True],
          [ True, False, False]],

         [[ True, False, False],
          [ True,  True,  True],
          [ True,  True,  True],
          [ True, False, False],
          [ True, False, False],
          [ True,  True,  True],
          [ True, False, False]],

         [[ True, False, False],
          [ True,  True,  True],
          [ True,  True,  True],
          [ True, False, False],
          [ True, False, False],
          [ True,  True,  True],
          [ True, False, False]]]]), 'type_ids': tensor([[[[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]],

         [[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]],

         [[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]],

         [[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]]]])}})
    """