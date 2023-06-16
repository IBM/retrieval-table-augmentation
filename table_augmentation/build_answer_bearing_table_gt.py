from util.line_corpus import jsonl_lines, write_open
import ujson as json
from util.args_help import fill_from_args
from collections import defaultdict
from table_augmentation.table import Table
from table_augmentation.augmentation_tasks import TaskOptions, Query, AugmentationTask


class Options:
    def __init__(self):
        self.table_corpus = ''
        self.query_tables = ''
        self.retrieval_gt = ''  # the output file
        self.task = TaskOptions()


if __name__ == '__main__':
    opts = Options()
    fill_from_args(opts)
    query_maker = opts.task.get_query_maker()

    # find the answer to question-id-list mapping
    answer2ids = defaultdict(list)
    normalize = opts.task.answer_normalization.get_normalizer()
    skip_count = 0
    for line in jsonl_lines(opts.query_tables):
        jobj = json.loads(line)
        table = Table.from_dict(jobj)
        queries = query_maker(table)
        if len(queries) == 0:
            skip_count += 1
            continue
        for query in queries:
            qid = query.qid
            for answer in query.answers:
                if len(answer) > 0:
                    answer2ids[answer].append(qid)
    if skip_count > 0:
        print(f'WARNING: Could not make query for {skip_count} query tables!')

    # for each table that is answer bearing for a question, give the list of questions it is answer-bearing for
    with write_open(opts.retrieval_gt) as out:
        for line in jsonl_lines(opts.table_corpus):
            jobj = json.loads(line)
            tbl = Table.from_dict(jobj)
            if opts.task.task in [AugmentationTask.row, AugmentationTask.cell]:
                found_answers = tbl.answer_bearing_cells(answer2ids.keys(), normalize)
            elif opts.task.task == AugmentationTask.col:
                found_answers = tbl.answer_bearing_header(answer2ids.keys(), normalize)
            else:
                raise NotImplementedError
            answers_qids = []
            for found in found_answers:
                answers_qids.extend(answer2ids[found])
            if len(answers_qids) > 0:
                out.write(json.dumps({'table_id': tbl.table_id, 'answer_bearing_for': list(set(answers_qids))})+'\n')

