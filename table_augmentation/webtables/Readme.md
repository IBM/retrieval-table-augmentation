## Download the 2015 English relational tables:

#### Install fast bzip2 if not present
```bash
sudo apt-get install -y lbzip2
```
You can also edit the download script to use plain bzip2

### Run the download script (this can take a while)
```bash
mkdir -p ${WEBTABLE_DIR}/webtables_orig
cd ${WEBTABLE_DIR}/webtables_orig
bash ${PYTHONPATH}/table_augmentation/webtables/webtable_download.sh
```

## Convert the format
```bash
python ${PYTHONPATH}/table_augmentation/webtables/webtables_to_jsonl.py \
--input ${WEBTABLE_DIR}/webtables_orig \
--output ${WEBTABLE_DIR}/webtables_clean \
--report ${WEBTABLE_DIR}/webtables_report.txt
```

## Split data
```bash
python ${PYTHONPATH}/table_augmentation/webtables/split_by_id.py \
--input ${WEBTABLE_DIR}/webtables_clean \
--output_dir ${WEBTABLE_DIR}/dataset
```

## Build passages
```bash
python ${PYTHONPATH}/table_augmentation/tables2passages.py \
--train_tables ${WEBTABLE_DIR}/dataset/train \
--passages ${WEBTABLE_DIR}/passages/col \
--task.task col --task.num_passage_rows 2

python ${PYTHONPATH}/table_augmentation/tables2passages.py \
--train_tables ${WEBTABLE_DIR}/dataset/train \
--passages ${WEBTABLE_DIR}/passages/row \
--task.task row --task.num_passage_rows 3
```

## Remaining steps are the same as with entitables, we document here for specific options used

### Index with BM25
```bash
python ${PYTHONPATH}/dpr/anserini_index.py \
--input ${WEBTABLE_DIR}/passages/row --replace_title_text_sep_in_title \
--output_dir ${WEBTABLE_DIR}/passages/row_anserini \
--jar ${YOUR_ANSERINI_DIR}/Anserini/target/anserini-0.4.1-SNAPSHOT-fatjar.jar

python ${PYTHONPATH}/dpr/anserini_index.py \
--input ${WEBTABLE_DIR}/passages/col --replace_title_text_sep_in_title \
--output_dir ${WEBTABLE_DIR}/passages/col_anserini \
--jar ${YOUR_ANSERINI_DIR}/Anserini/target/anserini-0.4.1-SNAPSHOT-fatjar.jar
```

### BM25 retrieval baseline
```bash
python ${PYTHONPATH}/dpr/table_aug_bm25_apply.py \
  --jar ${YOUR_ANSERINI_DIR}/Anserini/target/anserini-0.4.1-SNAPSHOT-fatjar.jar \
--task.task row --task.answer_normalization deunicode \
--tables ${WEBTABLE_DIR}/dataset/dev.jsonl.gz \
--output ${WEBTABLE_DIR}/apply/bm25/dev_rows.jsonl.gz \
--anserini_index ${WEBTABLE_DIR}/passages/row_anserini/index

python ${PYTHONPATH}/dpr/table_aug_bm25_apply.py \
  --jar ${YOUR_ANSERINI_DIR}/Anserini/target/anserini-0.4.1-SNAPSHOT-fatjar.jar \
--task.task col --task.answer_normalization deunicode \
--tables ${WEBTABLE_DIR}/dataset/dev.jsonl.gz \
--output ${WEBTABLE_DIR}/apply/bm25/dev_cols.jsonl.gz \
--anserini_index ${WEBTABLE_DIR}/passages/col_anserini/index
```

### Create DPR training data
```bash
python ${PYTHONPATH}/table_augmentation/table_dpr_bm25_answer_bearing.py \
  --train_file ${WEBTABLE_DIR}/dataset/train --task.task row --task.answer_normalization deunicode  \
  --jar ${YOUR_ANSERINI_DIR}/Anserini/target/anserini-0.4.1-SNAPSHOT-fatjar.jar \
  --anserini_index ${WEBTABLE_DIR}/passages/row_anserini/index  \
  --output_dir ${WEBTABLE_DIR}/dpr_train/row_qtt_a

python ${PYTHONPATH}/table_augmentation/table_dpr_bm25_answer_bearing.py \
  --train_file ${WEBTABLE_DIR}/dataset/train --task.task col --task.answer_normalization deunicode  \
  --jar ${YOUR_ANSERINI_DIR}/Anserini/target/anserini-0.4.1-SNAPSHOT-fatjar.jar \
  --anserini_index ${WEBTABLE_DIR}/passages/col_anserini/index  \
  --output_dir ${WEBTABLE_DIR}/dpr_train/col
```

### Train DPR
```bash
python ${PYTHONPATH}/midpr/biencoder_trainer.py \
--train_dir ${WEBTABLE_DIR}/dpr_train/row_qtt_a  \
--output_dir ${WEBTABLE_DIR}/models/dpr_e1_row_qtt_a  \
--seq_len_q  64 --seq_len_c  128 \
--num_train_epochs 1 \
--encoder_gpu_train_limit 16 \
--max_grad_norm 1.0 --learning_rate 5e-5 \
--full_train_batch_size 128

python ${PYTHONPATH}/midpr/biencoder_trainer.py \
--train_dir ${WEBTABLE_DIR}/dpr_train/col   --query_single_sequence \
--output_dir ${WEBTABLE_DIR}/models/dpr_e1_col  \
--seq_len_q  64 --seq_len_c  128 \
--num_train_epochs 1 \
--encoder_gpu_train_limit 16 \
--max_grad_norm 1.0 --learning_rate 5e-5 \
--full_train_batch_size 128
```

### Build DPR Index
(and same for col)
```bash
python ${PYTHONPATH}/dpr/index_simple_corpus.py \
--embed 1of2 --sharded_index \
--dpr_ctx_encoder_path ${WEBTABLE_DIR}/models/dpr_e1_row_qtt_a/ctx_encoder \
--corpus ${WEBTABLE_DIR}/passages/row \
--output_dir ${WEBTABLE_DIR}/passages/row_index

python ${PYTHONPATH}/dpr/index_simple_corpus.py \
--embed 2of2 --sharded_index \
--dpr_ctx_encoder_path ${WEBTABLE_DIR}/models/dpr_e1_row_qtt_a/ctx_encoder \
--corpus ${WEBTABLE_DIR}/passages/row  \
--output_dir ${WEBTABLE_DIR}/passages/row_index
```
#### And start the retrieval service(s)
You do not need both running at once.
```bash
python ${PYTHONPATH}/corpus/corpus_server_direct.py \
--port 5001 --corpus_dir ${WEBTABLE_DIR}/passages/row_index

python ${PYTHONPATH}/corpus/corpus_server_direct.py \
--port 5002 --corpus_dir ${WEBTABLE_DIR}/passages/col_index
```
You can now apply the DPR model to see retrieval improvement over BM25. See the main Readme.md.

### Train RATA
```bash
python ${PYTHONPATH}/extractive/raex_train.py \
--tables ${WEBTABLE_DIR}/dataset/train --task.task col --task.answer_normalization deunicode --train_instances 200000 \
--model_name_or_path bert-large-cased \
--dpr.qry_encoder_path ${WEBTABLE_DIR}/models/dpr_e1_col/qry_encoder \
--dpr.corpus_endpoint http://127.0.0.1:5002 --dpr.n_docs 5 \
--num_train_epochs 1 --warmup_fraction 0.1  --full_train_batch_size 32 \
--output_dir ${WEBTABLE_DIR}/models/raex_dpr_e1_col_n200_b32

python ${PYTHONPATH}/extractive/raex_train.py \
--tables ${WEBTABLE_DIR}/dataset/train --task.task row --task.answer_normalization deunicode --train_instances 200000 \
--model_name_or_path bert-large-cased \
--dpr.qry_encoder_path ${WEBTABLE_DIR}/models/dpr_e1_row_qtt_a/qry_encoder \
--dpr.corpus_endpoint http://127.0.0.1:5001 --dpr.n_docs 5 \
--num_train_epochs 1 --warmup_fraction 0.1  --full_train_batch_size 32 \
--output_dir ${WEBTABLE_DIR}/models/raex_dpr_e1_row_qtt_a_n200_b32

```

### Apply RATA
(same for test.jsonl.gz)
```bash
python ${PYTHONPATH}/extractive/raex_apply.py \
--tables ${WEBTABLE_DIR}/dataset/dev.jsonl.gz --task.task col --task.answer_normalization deunicode \
--model_name_or_path bert-large-cased --resume_from ${WEBTABLE_DIR}/models/raex_dpr_e1_col_n200_b32 \
--dpr.corpus_endpoint http://127.0.0.1:5002 --dpr.n_docs 10 \
--output_dir ${WEBTABLE_DIR}/apply/raex_dpr_e1_col_n200_b32_n10

python ${PYTHONPATH}/extractive/raex_apply.py \
--tables ${WEBTABLE_DIR}/dataset/dev.jsonl.gz --task.task row --task.answer_normalization deunicode \
--model_name_or_path bert-large-cased --resume_from ${WEBTABLE_DIR}/models/raex_dpr_e1_row_qtt_a_n200_b32 \
--dpr.corpus_endpoint http://127.0.0.1:5001 --dpr.n_docs 10 \
--output_dir ${WEBTABLE_DIR}/apply/raex_dpr_e1_row_qtt_a_n200_b32_n10
```