from util.args_help import fill_from_args
from util.line_corpus import jsonl_lines, shuffled_writer, write_open
import ujson as json
import random


class Options:
    def __init__(self):
        self.positive_bag_train_data = ''
        self.positive_pids = ''
        self.output_dir = ''
        self.__required_args__ = ['positive_pids', 'output_dir', 'positive_bag_train_data']


if __name__ == "__main__":
    opts = Options()
    fill_from_args(opts)
    rand = random.Random(123)
    with shuffled_writer(opts.output_dir) as writer, write_open(opts.positive_pids) as id2pids:
        for line in jsonl_lines(opts.positive_bag_train_data):
            jobj = json.loads(line)
            inst_id = jobj['id']
            positives = jobj['positives']
            negatives = jobj['negatives']
            id2pids.write(json.dumps({'id': inst_id, 'positive_pids': [p['pid'] for p in positives]})+'\n')
            for pndx, pos in enumerate(positives):
                if len(negatives) >= 2 * len(positives):
                    negs = negatives[pndx::len(positives)]
                else:
                    rand.shuffle(negatives)
                    negs = negatives[:3]
                writer.write(json.dumps({'id': jobj['id'], 'query': jobj['query'],
                                         'positive': pos, 'negatives': negs})+'\n')
