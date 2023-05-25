import numpy as np
import pandas as pd
import argparse
import json
from tqdm import tqdm
import pdb
from torch_util.line_corpus import jsonl_lines
import unicodedata

punct = '''!()-[]{};:'"\,<>./?@#%^&*_~'''

def clean_rows(data):
    return data.dropna(axis = 'index')

def clean_cols(data):
    return data.loc[:,~data.columns.isnull()]

def clean_headers(header_list):
    ascii_removed = [unicodedata.normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode('utf8') for text in header_list] # ASCII characters removed
    punct_removed = [''.join(['' if i in punct else i for i in text]) for text in ascii_removed] # Removed punctuation
    return punct_removed    
    
def clean_data(headers, rows, drop_rows = True, drop_cols = True):
    df = pd.DataFrame(rows, columns=headers)
    df = df.replace('', np.NaN)
    df = df.replace(' ', np.NaN)
    if drop_rows:
        df = clean_rows(df) # dropped rows which have any value as NULL
    if df.shape[0] == 0:
        return False, None, None
    if drop_cols:
        df = clean_cols(df)
    df = df.fillna('')    
    return True, df.values.tolist(), df.columns.tolist()

def main(hypers):
    num_instances = sum(1 for _ in jsonl_lines(hypers.filename))
    clean_file = []
    with tqdm(total=num_instances, desc='[CLEAN]', leave=True) as pbar:
        for line_ndx, line in enumerate(jsonl_lines(hypers.filename)):
            jobj = json.loads(line)
            keep_flag, jobj["rows"], jobj["header"] = clean_data(jobj["header"], jobj["rows"], hypers.drop_rows, hypers.drop_cols)
            if keep_flag:
                jobj["header"] = clean_headers(jobj["header"])
                clean_file.append(jobj)
            pbar.update(1)
    print("Initial number of tables:", str(num_instances))
    print("Final number of cleaned tables:", str(len(clean_file)))
    with open(hypers.output, 'w') as f:
        for item in clean_file:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Dataset Files")
    parser.add_argument("--data_dir", type=str, default="./data/queries/")
    parser.add_argument("--input", type=str, default="train.jsonl")
    parser.add_argument("--output", type=str, default="train_cleaned.jsonl")
    parser.add_argument("--drop_cols", help='Drop Columns with NULL headers', action='store_true', dest = 'drop_cols')
    parser.add_argument("--drop_rows", help='Drop row with any NULL value', action='store_true', dest = 'drop_rows')
    args = parser.parse_args()
    args.filename = args.data_dir+args.input
    args.output = args.data_dir+args.output    
    main(args)