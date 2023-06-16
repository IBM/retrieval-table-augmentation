from table_augmentation.explore.table_passages import table2query
from util.line_corpus import jsonl_lines, ShuffledWriter
from util.args_help import fill_from_args
import ujson as json


class Options:
    def __init__(self):
        self.tables = ''
        self.output_dir = ''
        self.num_seeds = 2
        self.row_or_column = 'row'
        self.__required_args__ = ['tables', 'output_dir']


opts = Options()
fill_from_args(opts)

writer = ShuffledWriter(opts.output_dir, num_files=8)
for line in jsonl_lines(opts.tables):
    jobj = json.loads(line)
    query = table2query(jobj, opts.num_seeds, opts.row_or_column)
    if query is None:
        # TODO: log
        continue
    # {'id':  'table_id':  'title': title, 'text': text, 'answers': answers}
    writer.write(json.dumps({'id': query['id'],
                             'input': query['title']+'\n\n'+query['text'],
                             'answers': query['answers']})+'\n')
writer.close()
