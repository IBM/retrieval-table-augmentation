from generative.rag_hypers import RagHypers
from torch_util.transformer_optimize import TransformerOptimize, LossHistory
from util.line_corpus import read_lines, block_shuffle, jsonl_lines
import ujson as json
import random
from generative.rag_util import prepare_seq2seq_batch_labels, prefered_answers, retrieved_docs_to_bart_input
from corpus.corpus_client import CorpusClient
import logging
import torch
import torch.nn.functional as F
from transformers import RagConfig
from dpr.dpr_util import queries_to_vectors
from table_augmentation.table import Table
from table_augmentation.augmentation_tasks import Query

logger = logging.getLogger(__name__)


class Options(RagHypers):
    def __init__(self):
        super().__init__()


hypers = Options()
hypers.fill_from_args()

tokenizer, model = hypers.get_tokenizer_and_model()
config = RagConfig.from_pretrained('facebook/rag-token-nq')

model = model.to(hypers.device)
model.train()
# construct rest retriever after the model
rest_retriever = CorpusClient(hypers)
optimizer = TransformerOptimize(hypers, hypers.num_train_epochs * hypers.num_instances, model)
loss_history = LossHistory(hypers.num_instances //
                           (hypers.full_train_batch_size // hypers.gradient_accumulation_steps))
query_maker = hypers.task.get_query_maker()
batch_count = 0
if hypers.n_gpu < 1:
    raise ValueError('Must have GPU')
# torch.autograd.set_detect_anomaly(True)


def retrieve(queries, id_batch):
    # CONSIDER: this could use a retriever_dpr (note it would split the optimizer though)
    query_vectors = queries_to_vectors(tokenizer, model.rag.question_encoder, queries,
                                       max_query_length=hypers.max_context_length)
    doc_scores, docs, doc_vectors = rest_retriever.retrieve(query_vectors, n_docs=hypers.n_docs,
                                                            exclude_by_pid_prefix=id_batch)
    context_input_ids, context_attention_mask = retrieved_docs_to_bart_input(config, hypers.max_context_length,
                                                                             tokenizer, queries, docs)
    return context_input_ids.reshape(len(queries) * hypers.n_docs, -1).to(model.device), \
           context_attention_mask.reshape(len(queries) * hypers.n_docs, -1).to(model.device), \
           doc_scores.reshape(len(queries), hypers.n_docs), docs


def one_batch(queries, answers, id_batch):
    global batch_count
    context_input_ids, context_attention_mask, doc_scores, docs = retrieve(queries, id_batch)

    labels = prepare_seq2seq_batch_labels(tokenizer, answers, return_tensors="pt",
                                          max_target_length=hypers.max_target_length).to(optimizer.hypers.device)

    outputs = optimizer.model(labels=labels,
                              context_input_ids=context_input_ids, context_attention_mask=context_attention_mask,
                              doc_scores=doc_scores)
    batch_count += 1
    loss = outputs.loss.mean()
    loss_history.note_loss(loss.item())
    optimizer.step_loss(loss,
                        retrieval_time=rest_retriever.retrieval_time/(batch_count * hypers.per_gpu_train_batch_size))


def train():
    rand = random.Random(hypers.seed)
    query_batch = []
    answer_batch = []
    id_batch = []
    skip_count = 0
    while True:
        optimizer.model.train()
        inst_count = 0
        for line in block_shuffle(read_lines(hypers.tables, shuffled_files=rand), rand=rand, block_size=100000):
            inst = json.loads(line)
            if hypers.is_query:
                queries = [Query.from_dict(inst)]
            else:
                queries = query_maker(Table.from_dict(inst))
            if len(queries) == 0:
                skip_count += 1
            for query in queries:
                inst_count += 1
                if inst_count % hypers.world_size != hypers.global_rank:
                    continue

                input_text = query.title + '\n\n' + query.text
                query_batch.append(input_text)
                answer_batch.append('; '.join(query.answers[:10]))  # NOTE: just the first 10 answers?
                id_batch.append(query.table_id)
                if len(query_batch) == hypers.per_gpu_train_batch_size * hypers.n_gpu:
                    one_batch(query_batch, answer_batch, id_batch)
                    if not optimizer.should_continue():
                        return
                    query_batch = []
                    answer_batch = []
                    id_batch = []
        print(f'skipped {skip_count}')


train()
if hypers.world_size > 1:
    torch.distributed.barrier()
if hypers.global_rank == 0:
    (optimizer.model.module if hasattr(optimizer.model, "module") else optimizer.model).save_pretrained(hypers.output_dir)
logger.info(f'loss_history = {loss_history.loss_history}')
hypers.cleanup_corpus_server()
