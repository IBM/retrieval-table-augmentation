from util.args_help import fill_from_args
from util.line_corpus import jsonl_lines, block_shuffle
from collections import Counter
import os
import ujson as json
import tabulate


class Options:
    def __init__(self):
        self.queries_dir = ''
        self.split_definitions = ''
        self.table_dir = ''
        self.train_limit = 100000
        self.first_col_ents = False
        self.any_cell_entity = False
        self.min_entity_train_count = 2
        self.num_seeds = 2
        self.min_rows = 4
        self.min_cols = 4
        self.allow_header_repeat = False
        self.train_limit_entities_only = False
        self.header_lower_case = False
        self.header_entity_normalize = False


def is_entity(cell):
    return cell.startswith('[') and cell.endswith(']') and cell.find('|') >= 0 and len(cell[1:-1].split('|')) == 2


def cell_text(cell):
    if is_entity(cell):
        page, label = cell[1:-1].split('|')
        page = page.replace('_', ' ')
        return page  # the point of EntiTables is we predict the page
    return cell


opts = Options()
fill_from_args(opts)

header2count = Counter()
entity2count = Counter()


def header_normalize(head: str) -> str:
    if opts.header_entity_normalize:
        head = cell_text(head)
    return head.lower() if opts.header_lower_case else head


# iterate over the train tables filtered as specified in the options
#  build up the header2count and entity2count

test_tables = dict()  # id2split
for file in os.listdir(opts.split_definitions):
    with open(os.path.join(opts.split_definitions, file)) as f:
        test_table_list = json.load(f)
        for table_id in test_table_list:
            test_tables[table_id] = file[:-5]


def table_generator(table_dir):
    for file in os.listdir(table_dir):
        with open(os.path.join(table_dir, file)) as f:
            tables = json.load(f)
        for table_id, table in tables.items():
            yield table_id, table


table_count = 0
for table_id, table in block_shuffle(table_generator(opts.table_dir)):
    if not opts.train_limit_entities_only and opts.train_limit > 0 and table_count >= opts.train_limit:
        break
    if table_id in test_tables:
        continue
    header = [header_normalize(h) for h in table['title']]
    title = table['pgTitle'] + '\n' + table['caption']
    assert all([len(row) == len(header) for row in table['data']])

    if len(header) < opts.min_cols:
        continue
    if len(table['data']) < opts.min_rows:
        continue
    if not opts.allow_header_repeat and any([h == header[0] for h in header[1:]]):
        continue

    table_count += 1
    for h in header:
        header2count[h] += 1
    if opts.train_limit < 0 or table_count <= opts.train_limit:
        for row in table['data']:
            # maybe this version?
            if opts.first_col_ents:
                if is_entity(row[0]):
                    entity2count[cell_text(row[0])] += 1
            elif opts.any_cell_entity:
                for cell in row:
                    entity2count[cell_text(cell)] += 1
            else:
                for cell in row:
                    if is_entity(cell):
                        entity2count[cell_text(cell)] += 1
print(f'Filtered table count = {table_count}')

for col_filename in ['column_id_validation.jsonl', 'column_id_test.jsonl']:
    in_recall = 0
    out_recall = 0
    instance_count = 0
    sum_recall_ceil = 0
    out_recall_set = set()
    in_recall_set = set()
    for line in jsonl_lines(os.path.join(opts.queries_dir, col_filename)):
        jobj = json.loads(line)
        header = jobj['header']
        cur_in_recall = 0
        cur_out_recall = 0
        for hndx in range(opts.num_seeds, len(header)):
            h = header_normalize(header[hndx])
            if h in header2count:
                cur_in_recall += 1
                in_recall_set.add(h)
            else:
                cur_out_recall += 1
                out_recall_set.add(h)
        in_recall += cur_in_recall
        out_recall += cur_out_recall
        instance_count += 1
        sum_recall_ceil += cur_in_recall / (cur_in_recall + cur_out_recall)
    print(f'For {col_filename}')
    print(tabulate.tabulate([['Train header set', len(header2count)],
                             ['In recall set', len(in_recall_set)],
                             ['Out recall set', len(out_recall_set)],
                             ['Recall fraction', in_recall/(in_recall+out_recall)],
                             ['Recall ceiling', sum_recall_ceil/instance_count]]))


for row_filename in ['row_id_validation.jsonl', 'row_id_test.jsonl']:
    in_recall = 0
    out_recall = 0
    instance_count = 0
    sum_recall_ceil = 0
    out_recall_set = set()
    in_recall_set = set()
    for line in jsonl_lines(os.path.join(opts.queries_dir, row_filename)):
        jobj = json.loads(line)
        rows = jobj['rows']
        cur_in_recall = 0
        cur_out_recall = 0
        for rndx, is_ent in enumerate(jobj['row_entities']):
            if rndx < opts.num_seeds:
                continue
            assert is_ent
            row = rows[rndx]
            if row[0] in entity2count and entity2count[row[0]] >= opts.min_entity_train_count:
                cur_in_recall += 1
                in_recall_set.add(row[0])
            else:
                cur_out_recall += 1
                out_recall_set.add(row[0])
        in_recall += cur_in_recall
        out_recall += cur_out_recall
        instance_count += 1
        sum_recall_ceil += cur_in_recall / (cur_in_recall + cur_out_recall)
    print(f'For {row_filename}')
    min_count_entity_set_size = 0
    for e, c in entity2count.items():
        if c >= opts.min_entity_train_count:
            min_count_entity_set_size += 1
    print(tabulate.tabulate([['Train entity set', len(entity2count)],
                             [f'Train entity set (count >= {opts.min_entity_train_count})', min_count_entity_set_size],
                             ['In recall set', len(in_recall_set)],
                             ['Out recall set', len(out_recall_set)],
                             ['Recall fraction', in_recall/(in_recall+out_recall)],
                             ['Recall ceiling', sum_recall_ceil/instance_count]]))
