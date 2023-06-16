from util.args_help import fill_from_args
from util.line_corpus import jsonl_lines, block_shuffle
import ujson as json
from table_augmentation.table import Table
from table_augmentation.augmentation_tasks import TaskOptions, Passage, Query


class Options:
    def __init__(self):
        self.tables = ''
        self.apply_output = ''
        self.passage_limit = 5
        self.answer_limit = 10
        self.task = TaskOptions()


opts = Options()
fill_from_args(opts)

id2table = dict()
for line in jsonl_lines(opts.tables):
    jobj = json.loads(line)
    table = Table.from_dict(jobj)
    id2table[table.table_id] = table

tabulate_passage = opts.task.get_passage_displayer()
for line in block_shuffle(jsonl_lines(opts.apply_output)):
    print('\n\n')
    jobj = json.loads(line)
    table_id = jobj['table_id']
    if table_id not in id2table:
        print(f'Missing table: {table_id}')  # FIXME: how can this happen?
        continue
    query_table = id2table[table_id]
    print(query_table.tabulate())
    query_title, query_text = jobj['query']
    gt_answers = jobj['answers'][:opts.answer_limit]
    # print(query_title+'\n'+query_text)
    print('ANSWERS: '+str(gt_answers))
    print('PASSAGES:\n' + ('*'*80) + '\n')
    passages = jobj['passages'][:opts.passage_limit]
    for passage_dict in passages:
        passage = Passage.from_dict(passage_dict)
        print(tabulate_passage(passage))
        print('')
    if 'aggregated_predictions' in jobj:
        pred_answers = jobj['aggregated_predictions']
        print(pred_answers)
    k = input('...')
