from util.args_help import fill_from_args
from util.line_corpus import jsonl_lines
import ujson as json
from collections import Counter
import numpy as np
import os


class Options:
    def __init__(self):
        self.queries = ''


def show_distribution(item_counts: Counter, thresholds=(1, 2, 3, 5, 10, 50, 100, 1000)):
    """

    :param item_counts: counts of items we wish to see the distribution of
    :param thresholds: count thresholds of interest, must all be greater than 0
    :return: nothing, we print
    """
    thresholds = list(thresholds)
    thresholds.sort()
    assert all([t > 0 for t in thresholds])
    assert all([thresholds[ndx+1] > thresholds[ndx] for ndx in range(len(thresholds)-1)])
    counts = np.zeros(len(thresholds) + 1)
    sums = np.zeros(len(thresholds) + 1)
    samples = [[] for _ in range(len(thresholds) + 1)]
    total = 0
    for h, count in item_counts.items():
        bucket_ndx = len(thresholds)
        for tndx in range(len(thresholds)):
            if count <= thresholds[tndx]:
                bucket_ndx = tndx
                break
        sums[bucket_ndx] += count
        total += count
        counts[bucket_ndx] += 1
        sample_list = samples[bucket_ndx]
        if len(sample_list) < 5:
            sample_list.append(f'{h}@{count}')

    print(f'Set size per occurrence count:')
    for tndx in range(len(thresholds)):
        inequality = '==' if thresholds[tndx] == 1 else '<='
        print(f'  {inequality} {thresholds[tndx]}: {counts[tndx]} ex: {samples[tndx]}')
    print(f'  >{thresholds[len(thresholds) - 1]}: {counts[len(thresholds)]} ex: {samples[len(thresholds)]}')

    print(f'Sums per occurrence count:')
    for tndx in range(len(thresholds)):
        inequality = '==' if thresholds[tndx] == 1 else '<='
        print(f'  {inequality} {thresholds[tndx]}: {sums[tndx]} {sums[tndx] / total}')
    print(f'  > {thresholds[len(thresholds) - 1]}: {sums[len(thresholds)]} fraction: {sums[len(thresholds)] / total}')


def header_and_entity_distributions(queries_dir):
    header_counts = Counter()
    ent_counts = Counter()
    for line in jsonl_lines(queries_dir):
        jobj = json.loads(line)
        for h in jobj['header']:
            header_counts[h.lower().strip()] += 1  # CONSIDER: what normalization is appropriate here?
        for row, is_ent in zip(jobj['rows'], jobj['row_entities']):
            if is_ent:
                ent_counts[row[0]] += 1

    print(f'Number of headers = {len(header_counts)}')
    show_distribution(header_counts)
    print(f'Number of entities = {len(ent_counts)}')
    show_distribution(ent_counts)


def present_in_train(queries_dir):
    """
    Headers in train 0.9544479775197812, distinct out of train 612
    Entities in train 0.8084802725473045, distinct out of train 8078

    column_id_test.jsonl.gz        row_id_test.jsonl.gz        train.jsonl.gz
    column_id_validation.jsonl.gz  row_id_validation.jsonl.gz
    :param queries_dir:
    :return:
    """
    train = os.path.join(queries_dir, 'train.jsonl.gz')
    headers = set()
    ents = set()
    for line in jsonl_lines(train):
        jobj = json.loads(line)
        for h in jobj['header']:
            headers.add(h.lower().strip())
        for row, is_ent in zip(jobj['rows'], jobj['row_entities']):
            if is_ent:
                ents.add(row[0])

    out_of_train = set()
    in_train_count = 0
    out_train_count = 0
    for line in jsonl_lines([os.path.join(queries_dir, 'column_id_test.jsonl.gz'), os.path.join(queries_dir, 'column_id_validation.jsonl.gz')]):
        jobj = json.loads(line)
        for h in jobj['header']:
            if h.lower().strip() not in headers:
                out_of_train.add(h.lower().strip())
                out_train_count += 1
            else:
                in_train_count += 1

    print(f'Headers in train {in_train_count/(in_train_count+out_train_count)}, distinct out of train {len(out_of_train)}')

    out_of_train = set()
    in_train_count = 0
    out_train_count = 0
    for line in jsonl_lines([os.path.join(queries_dir, 'row_id_test.jsonl.gz'), os.path.join(queries_dir, 'row_id_validation.jsonl.gz')]):
        jobj = json.loads(line)
        for row, is_ent in zip(jobj['rows'], jobj['row_entities']):
            if is_ent:
                if row[0] not in ents:
                    out_of_train.add(row[0])
                    out_train_count += 1
                else:
                    in_train_count += 1
    print(f'Entities in train {in_train_count / (in_train_count + out_train_count)}, distinct out of train {len(out_of_train)}')


if __name__ == "__main__":
    opts = Options()
    fill_from_args(opts)
    present_in_train(opts.queries)
"""
python table_augmentation/entitable_stats.py --queries /users/mrglass/Box\ Sync/Table\ Augmentation/queries/train.jsonl.gz:/users/mrglass/Box\ Sync/Table\ Augmentation/queries/column_id_validation.jsonl.gz:/users/mrglass/Box\ Sync/Table\ Augmentation/queries/column_id_test.jsonl.gz     
Number of headers = 59684
<= 1: 33800.0 ex: ['alan oppenheimer@1', 'ann jillian@1', 'ann morgan guilbert@1', "annette o'toole@1", 'arte johnson@1']
<= 2: 8345.0 ex: ['highest level@2', 'post-season series win drought@2', '# of seats total@2', '# of national votes@2', '% of national vote@2']
<= 3: 3692.0 ex: ['country/agency@3', 'date of landing/impact@3', 'elec- tion@3', 'republican nominee@3', 'measures@3']
<= 5: 4084.0 ex: ['airline (in arabic)@5', 'city/region@5', 'democratic nominee@4', 'juventud de las piedras@5', '%w@5']
<= 10: 3776.0 ex: ['electorates of the australian capital territory@7', 'sweet sixteen@7', 'elite eight@7', 'final four@9', '2008–09 season@9']
<= 50: 4239.0 ex: ['color@45', 'history of sheffield wednesday f.c.@35', 'rotherham county f.c.@19', 'south shields f.c. (1889)@15', 'aberdeen f.c.@43']
<= 100: 820.0 ex: ['term in office@92', 'bury f.c.@96', 'cardiff city f.c.@67', 'huddersfield town a.f.c.@82', 'portsmouth f.c.@78']
 >100: 928.0 ex: ['season@11466', 'age@916', 'overall@1764', 'slalom@218', 'giant slalom@112']
<= 1: 33800.0 0.03161233180228638
<= 2: 16690.0 0.01560975792248993
<= 3: 11076.0 0.01035911795982615
<= 5: 17940.0 0.01677885303352123
<= 10: 28571.0 0.026721773133820237
<= 50: 90407.0 0.08455550536240546
<= 100: 57512.0 0.05378959842050574
 >100: 813207.0 ex: 0.7605730623651449
"""