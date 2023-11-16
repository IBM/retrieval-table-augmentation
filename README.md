# This is the code for reproducing our paper: "Retrieval-Based Transformer for Table Augmentation"


## Retrieval

### 1) first create passages from your table corpus, this may involve splitting tables into multiple passages
For example:
```bash
python ${PYTHONPATH}/table_augmentation/entitables/convert_entitables.py \
--table_dir ${TBL_DIR}/tables_redi2_1 \
--split_definitions ${TBL_DIR}/sigir2017-table/Data \
--passage_dir ${TBL_DIR}/passages \
--query_dir ${TBL_DIR}/queries
```

These passages should be in jsonl files with 'pid', 'title' and 'text' fields.
The table_id that the passage comes from should be a prefix of the pid. This will allow excluding by pid prefix during training.

### 2) index your table passages with Anserini

Download and build [Anserini](https://github.com/castorini/anserini). 
You will need to have [Maven](https://maven.apache.org/index.html) and a [Java JDK](https://jdk.java.net/).
```bash
git clone https://github.com/castorini/anserini.git
cd anserini
# to use the 0.4.1 version dprBM25.jar is built for
git checkout 3a60106fdc83473d147218d78ae7dca7c3b6d47c
export JAVA_HOME=your JDK directory
mvn clean package appassembler:assemble
```

Run formating and indexing. For example:
```bash
python ${PYTHONPATH}/dpr/anserini_index.py \
--jar ${YOUR_ANSERINI_DIR}/Anserini/target/anserini-0.4.1-SNAPSHOT-fatjar.jar \
--input ${TBL_DIR}/passages/row/a.jsonl.gz \
--output_dir ${TBL_DIR}/passages/row

```

### 3) Create DPR training data

For example:
```bash
python ${PYTHONPATH}/table_augmentation/table_dpr_bm25_answer_bearing.py \
  --train_file ${TBL_DIR}/queries/row_train.jsonl.gz --task.task row --task.answer_normalization identity \
  --jar ${YOUR_ANSERINI_DIR}/Anserini/target/anserini-0.4.1-SNAPSHOT-fatjar.jar \
  --anserini_index ${TBL_DIR}/passages/row/index \
  --output_dir ${TBL_DIR}/dpr_train/row
```

### 4) Train DPR

For example:
```bash
python ${PYTHONPATH}/midpr/biencoder_trainer.py \
--train_dir ${TBL_DIR}/dpr_train/row  \
--output_dir ${TBL_DIR}/models/dpr_e3_row  \
--seq_len_q  64 --seq_len_c  128 \
--num_train_epochs 3 \
--encoder_gpu_train_limit 16 \
--max_grad_norm 1.0 --learning_rate 5e-5 \
--full_train_batch_size 128
```

### 5) Build DPR index

This example shows building the index in two parts (if you want to use 2 GPUs in parallel)
```bash
python ${PYTHONPATH}/dpr/index_simple_corpus.py \
--embed 1of2 --sharded_index \
--dpr_ctx_encoder_path ${TBL_DIR}/models/dpr_e3_row/ctx_encoder \
--corpus ${TBL_DIR}/passages/row/a.jsonl.gz  \
--output_dir ${TBL_DIR}/passages/row/dpr_index

python ${PYTHONPATH}/dpr/index_simple_corpus.py \
--embed 2of2 --sharded_index \
--dpr_ctx_encoder_path ${TBL_DIR}/models/dpr_e3_row/ctx_encoder \
--corpus ${TBL_DIR}/passages/row/a.jsonl.gz \
--output_dir ${TBL_DIR}/passages/row/dpr_index
```

### 6) Apply DPR

For example:
```bash
python ${PYTHONPATH}/dpr/table_aug_dpr_apply.py \
--tables ${TBL_DIR}/queries/row_id_validation.jsonl --task.task row --task.answer_normalization identity \
--qry_encoder_path ${TBL_DIR}/models/dpr_e3_row/qry_encoder \
--corpus_endpoint ${TBL_DIR}/passages/row/dpr_index \
--output ${TBL_DIR}/apply/dpr_row.jsonl
```

Note that when a directory (rather than a http://IP:port) is provided as the corpus_endpoint, a DPR index service will be started.
An abnormal exit from either training or apply that uses such a service can leave the service still running.
Check to ensure that there are no stray DPR services left running with:
```bash
ps -ef | grep python
```
And check for any left over processes


## Reader

### 1) Train the row or column population model

```bash
export CORPUS=${TBL_DIR}/passages/row/dpr_index
```

#### Optionally start the corpus service
```bash
python ${PYTHONPATH}/corpus/corpus_server_direct.py \
--port 5001 --corpus_dir ${CORPUS}

export CORPUS_ENDPOINT=http://127.0.0.1:5001
```
#### OR let the train / apply scripts start it for you
```bash
export CORPUS_ENDPOINT=${CORPUS}
```

```bash
python ${PYTHONPATH}/extractive/raex_train.py \
--task.task row --task.answer_normalization identity \
--train_data ${TBL_DIR}/queries/row_train.jsonl.gz \
--model_name_or_path bert-large-cased \
--dpr.qry_encoder_path ${TBL_DIR}/models/dpr_e3_row/qry_encoder \
--dpr.corpus_endpoint ${CORPUS_ENDPOINT} --dpr.n_docs 5 \
--num_train_epochs 1 --warmup_fraction 0.1  --full_train_batch_size 32 \
--output_dir ${TBL_DIR}/models/raex_row
```

### 2) Apply the model

```bash
python ${PYTHONPATH}/extractive/raex_apply.py \
--tables ${TBL_DIR}/queries/row_id_validation.jsonl \
--task.task row --task.answer_normalization identity \
--model_name_or_path bert-large-cased --resume_from ${TBL_DIR}/models/raex_row \
--dpr.corpus_endpoint ${CORPUS_ENDPOINT} --dpr.n_docs 5 \
--output_dir ${TBL_DIR}/apply/raex_row
```

### NOTE: cell population
Cell filling is documented under table_augmentation/cell_filling

## Citation
```
@inproceedings{glass2023retrieval,
  title={Retrieval-Based Transformer for Table Augmentation},
  author={Glass, Michael and Wu, Xuecheng and Naik, Ankita and Rossiello, Gaetano and Gliozzo, Alfio},
  booktitle={Annual Meeting of the Association for Computational Linguistics},
  year={2023}
}
```
