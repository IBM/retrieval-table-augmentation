import sys

import numpy as np


_, inp, oup = sys.argv

with open(inp) as f:
    lines = f.readlines()
lines = [x.strip() for x in lines if x.startswith("H-")]

ids = [int(x.split("H-")[1].split()[0]) for x in lines]
assert len(ids) % (max(ids) + 1) == 0
assert np.all(np.array(sorted(ids)) ==
              np.arange(max(ids) + 1).reshape(-1, 1).repeat(len(ids) // (max(ids) + 1), 1).reshape(-1))
texts = [x.split("\t")[-1] for x in lines]

with open(oup, 'w') as f:
    for i in np.lexsort((np.arange(len(ids)), ids)):
        # preserve order for multi beam
        # can also use `kind='stable'`; but don't wanna do that
        f.write(texts[i] + '\n')
