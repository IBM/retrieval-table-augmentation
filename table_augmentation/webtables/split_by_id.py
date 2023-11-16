from util.args_help import fill_from_args
from util.line_corpus import jsonl_lines, write_open, read_lines, ShuffledWriter
import ujson as json
import os
from collections import Counter


class Options:
    def __init__(self):
        self.input = ''
        self.output_dir = ''
        self.id_list_dir = ''
        self._required_args = ['input', 'output_dir']


if __name__ == '__main__':
    opts = Options()
    fill_from_args(opts)
    if not opts.id_list_dir:
        opts.id_list_dir = os.path.dirname(__file__)
    split_map = {}
    for split in ['train', 'dev', 'test']:
        ids = set()
        for line in read_lines(os.path.join(opts.id_list_dir, f'{split}_ids.txt.bz2')):
            ids.add(line.strip())
        if split == 'train':
            out = ShuffledWriter(os.path.join(opts.output_dir, 'train'))
        else:
            out = write_open(os.path.join(opts.output_dir, f'{split}.jsonl.gz'))
        split_map[split] = (ids, out)
    skip_count = 0
    split_counts = Counter()
    for line in jsonl_lines(opts.input):
        jobj = json.loads(line)
        wrote_it = False
        for split in split_map.keys():
            ids, out = split_map[split]
            if jobj['table_id'] in ids:
                out.write(line)
                wrote_it = True
                split_counts[split] += 1
                break
        if not wrote_it:
            skip_count += 1
    for split in split_map.keys():
        ids, out = split_map[split]
        out.close()
        if len(ids) != split_counts[split]:
            print(f'WARNING: wrote only {split_counts[split]} for {split}, but {len(ids)} should be there!')
    print(f'Wrote: {split_counts}')
    print(f'Skipped {skip_count}')
