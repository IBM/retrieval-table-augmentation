import json
import os.path
import random
import sys
from collections import defaultdict

from table_augmentation.cell_filling.utils import get_type
from util.line_corpus import jsonl_lines, write_open

if __name__ == "__main__":
    _, inp, oup = sys.argv
    N = 1000
    assert not os.path.exists(oup)

    test = defaultdict(list)
    for line in jsonl_lines(inp):
        line = json.loads(line)
        line_type = get_type(line['answers'][0])
        test[line_type].append(line)

    with write_open(oup) as f:
        for t, lines in test.items():
            assert len(lines) >= N
            random.shuffle(lines)
            print("Type %s: total %d, downsample into %d (from %d tables)" % (
                t, len(lines), N, len(set([line['table']['table_id'] for line in lines[:N]]))
            ))
            for line in lines[:N]:
                line['type'] = t
                f.write(json.dumps(line) + '\n')
