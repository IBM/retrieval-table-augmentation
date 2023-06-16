from util.args_help import fill_from_args
from util.line_corpus import jsonl_lines, write_open
import ujson as json
from collections import Counter


class Options:
    def __init__(self):
        self.queries = ''
        self.entities_out = ''
        self.min_count = 2


opts = Options()
fill_from_args(opts)

# we filter entities like TABBIE (at least 2 occurrences)
#  "Our label space consists of 300K entities that occur at least twice in Wikipedia tables"
#  I get 274644 from this

ent_counts = Counter()
for line in jsonl_lines(opts.queries):
    jobj = json.loads(line)
    for row, is_ent in zip(jobj['rows'], jobj['row_entities']):
        if is_ent:
            ent_counts[row[0]] += 1

with write_open(opts.entities_out) as f:
    for ent, count in ent_counts.items():
        if count >= opts.min_count:
            if '\n' in ent:
                raise ValueError(f'ERROR: {ent}')
            f.write(ent+'\n')