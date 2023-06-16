import torch
from util.line_corpus import read_lines, write_open, jsonl_lines
import ujson as json
import numpy as np
import functools
from generative.rag_hypers import RagHypers
from util.reporting import Reporting
from dpr.dpr_util import queries_to_vectors
import logging
from corpus.corpus_client import CorpusClient
from generative.rag_util import retrieved_docs_to_bart_input
from transformers import RagConfig
import os
from table_augmentation.table import Table
from table_augmentation.augmentation_tasks import Query

logger = logging.getLogger(__name__)


class Options(RagHypers):
    def __init__(self):
        super().__init__()
        self.num_beams = 4
        self.num_return_sequences = 4
        self.n_docs_for_provenance = 20  # we'll supply this many document ids for reporting provenance
        self.retrieve_batch_size = 8
        self.num_instances = 0  # no training instances
        self.min_target_length = 2
        self.max_target_length = 128  # was 64
        self.length_penalty = 1.0
        # self.batch_size = 8

    def _post_argparse(self):
        self.num_beams = max(self.num_beams, self.num_return_sequences)


opts = Options()
opts.fill_from_args()
torch.set_grad_enabled(False)

tokenizer, model = opts.get_tokenizer_and_model()
config = RagConfig.from_pretrained('facebook/rag-token-nq')
query_maker = opts.task.get_query_maker()
normalize_answer = opts.task.answer_normalization.get_normalizer()

model = model.to(opts.device)
model.eval()
# construct rest retriever after the model
rest_retriever = CorpusClient(opts)
report = Reporting()
sum_rr = 0
count = 0


def retrieve(queries, id_batch):
    with torch.no_grad():
        query_vectors = queries_to_vectors(tokenizer, model.rag.question_encoder, queries,
                                           max_query_length=opts.max_context_length)

    # retrieve support docs
    doc_scores, docs, doc_vectors = rest_retriever.retrieve(query_vectors,
                                                            exclude_by_pid_prefix=id_batch,
                                                            n_docs=opts.n_docs,
                                                            n_docs_for_provenance=opts.n_docs_for_provenance)
    context_input_ids, context_attention_mask = retrieved_docs_to_bart_input(config, opts.max_context_length,
                                                                             tokenizer, queries, docs)
    retrieved_doc_ids = [dd['pid'] for dd in docs]
    return context_input_ids.reshape(len(queries), opts.n_docs, -1).to(model.device), \
           context_attention_mask.reshape(len(queries), opts.n_docs, -1).to(model.device), \
           doc_scores.reshape(len(queries), opts.n_docs), retrieved_doc_ids


def generate_one_instance(context_input_ids, context_attention_mask, doc_scores):
    """
    :param context_input_ids: n_docs x seq_len
    :param context_attention_mask: n_docs x seq_len
    :param doc_scores: n_docs
    :return:
    """
    # CONSIDER: try leading space for query too?
    with torch.no_grad():
        num_return_sequences = opts.num_return_sequences
        # because it runs out of memory if there are too many
        if num_return_sequences > 16:
            num_return_sequences = 16
        # generate answers
        beam_search_output = model.generate(
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            num_beams=max(int(1.5 * num_return_sequences), opts.num_beams),
            num_return_sequences=num_return_sequences,
            min_length=opts.min_target_length,
            max_length=opts.max_target_length,
            length_penalty=opts.length_penalty,
            return_dict_in_generate=True, output_scores=True
        )
        # BeamSearchDecoderOnlyOutput: sequences, sequences_scores
        generated_ids = beam_search_output.sequences.detach().cpu().numpy()
        if hasattr(beam_search_output, 'sequences_scores') and beam_search_output.sequences_scores is not None:
            generated_scores = beam_search_output.sequences_scores.detach().cpu().numpy()
        else:
            generated_scores = np.zeros(generated_ids.shape[0], dtype=np.float)

        answer_strings = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return answer_strings, generated_scores.tolist()


def record_one_instance(output, inst_id, input_text, answers, pred_text, pred_scores, doc_ids):
    global sum_rr
    global count
    pred_record = {'id': inst_id, 'input': input_text,
                   'predictions': pred_text, 'predictions_scores': pred_scores, 'doc_ids': doc_ids}
    if answers:
        pred_record['answers'] = answers

    rank = float("Inf")
    if answers:
        # each returned sequence is a '; ' separated list
        norm_pred_text = [normalize_answer(a) for beam in pred_text for a in beam.split('; ')]
        norm_answers = [normalize_answer(a) for a in answers]
        for ans in norm_answers:
            try:
                ndx = norm_pred_text.index(ans)
                if ndx + 1 < rank:
                    rank = ndx + 1
            except ValueError:
                pass
        sum_rr += 1.0 / rank
        count += 1
    output.write(json.dumps(pred_record) + '\n')
    if report.is_time():
        metrics = f' MRR = {sum_rr/count}' if count > 0 else ''
        print(f'Finished instance {report.check_count}; {report.check_count/report.elapsed_seconds()} per second. '
              f'{1000 * rest_retriever.retrieval_time/report.check_count} msecs per retrieval.{metrics}')


def one_batch(id_batch, query_batch, answer_batch, output):
    context_input_ids, context_attention_mask, doc_scores, retrieved_doc_ids = retrieve(query_batch, id_batch)
    # print(f'retrieved shapes: {context_input_ids.shape}, {context_attention_mask.shape}, {doc_scores.shape}, {retrieved_doc_ids}')
    for bi in range(len(query_batch)):
        context_ids_i = context_input_ids[bi]
        answer_strings, answer_scores = generate_one_instance(context_ids_i,
                                                              context_attention_mask[bi], doc_scores[bi:bi+1])
        record_one_instance(output, id_batch[bi], query_batch[bi], answer_batch[bi],
                            answer_strings, answer_scores, retrieved_doc_ids[bi])


inst_count = 0
with write_open(os.path.join(opts.output_dir, 'detailed', f'{opts.global_rank}.jsonl')) as output:
    id_batch, query_batch, answer_batch = [], [], []
    for line in read_lines(opts.tables):
        inst = json.loads(line)
        if opts.is_query:
            queries = [Query.from_dict(inst)]
        else:
            queries = query_maker(Table.from_dict(inst))
        for query in queries:
            inst_count += 1
            # skip line_ndx not for our rank
            if inst_count % opts.world_size != opts.global_rank:
                continue

            id_batch.append(query.table_id)
            query_batch.append(query.title + '\n\n' + query.text)
            answer_batch.append(query.answers)

            if len(query_batch) == opts.retrieve_batch_size:
                one_batch(id_batch, query_batch, answer_batch, output)
                id_batch, query_batch, answer_batch = [], [], []
    if len(query_batch) > 0:
        one_batch(id_batch, query_batch, answer_batch, output)
    metrics = f' MRR = {sum_rr/count}' if count > 0 else ''
    print(f'Finished instance {report.check_count}; {report.check_count/report.elapsed_seconds()} per second. '
          f'{1000 * rest_retriever.retrieval_time/report.check_count} msecs per retrieval.{metrics}')

if opts.world_size > 1:
    torch.distributed.barrier()  # wait for all to complete

if opts.global_rank == 0:
    # FIXME: evaluate
    pass

opts.cleanup_corpus_server()
