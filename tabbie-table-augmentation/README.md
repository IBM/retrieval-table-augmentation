# Tabbie: Tabular Information Embedding
This repository includes scripts for Tabbie(Tabular Information Embedding) model. 
The link to the paper is as follows.
https://arxiv.org/pdf/2105.02584.pdf

## Environment setup
```bash
conda create --name tabbie python=3.7
conda activate tabbie
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
pip install allennlp==1.1.0
pip install pytorch_pretrained_bert==0.6.2
pip install chardet pandas numpy tqdm cython pyyaml jinja2
```

## Commands for Column Population task

```commandline
python train_cp.py --train_table ./data/queries/train_header_cleaned.jsonl --n_seed_cols 1 --test_table ./data/queries/column_id_validation_header_cleaned.jsonl --num_train_epochs 3 --output_dir ./output/seed1/header_cleaned/ --model_file ./model/model_named.pt --saved_table_cls_dir ./data

python train_cp.py --train_table ./data/queries/train_header_cleaned.jsonl --n_seed_cols 2 --test_table ./data/queries/column_id_validation_header_cleaned.jsonl --num_train_epochs 3 --output_dir ./output/seed2/header_cleaned/ --model_file ./model/model_named.pt --saved_table_cls_dir ./data

python train_cp.py --train_table ./data/queries/train_header_cleaned.jsonl --n_seed_cols 3 --test_table ./data/queries/column_id_validation_header_cleaned.jsonl --num_train_epochs 3 --output_dir ./output/seed3/header_cleaned/ --model_file ./model/model_named.pt --saved_table_cls_dir ./data
```



