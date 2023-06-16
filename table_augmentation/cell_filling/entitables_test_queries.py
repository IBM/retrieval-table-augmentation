from util.args_help import fill_from_args
from util.line_corpus import jsonl_lines, write_open
import ujson as json

# from query:title, query:text, original_answers
# to: qid, table_id, title, text, answers


class Options:
    def __init__(self):
        self.test_tables = ''
        self.test_queries = ''


if __name__ == '__main__':
    opts = Options()
    fill_from_args(opts)
    with write_open(opts.test_queries) as out:
        for line in jsonl_lines(opts.test_tables):
            jobj = json.loads(line)
            qid = jobj['id']
            table_id = qid[:qid.find('[')-1]
            title = jobj['query']['title']
            text = jobj['query']['text']
            answers = jobj['original_answers']
            out.write(json.dumps({'qid': qid, 'table_id': table_id,
                                  'title': title, 'text': text, 'answers': answers})+'\n')
