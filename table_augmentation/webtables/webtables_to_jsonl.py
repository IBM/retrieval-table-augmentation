import os.path

from util.args_help import fill_from_args
from util.line_corpus import write_open, expand_files, shuffled_writer
import ujson as json
from collections import Counter
from table_augmentation.table import Table
from table_augmentation.webtables.filters import filter
from table_augmentation.webtables.cleaners import clean
from util.reporting import Reporting
import logging
import tarfile

logger = logging.getLogger(__name__)


class Options:
    def __init__(self):
        self.input = ''
        self.output = ''
        self.report = ''
        self.min_rows = 4
        self.min_cols = 4
        self.__required_args__ = ['input', 'output', 'report']


def json_from_tars(tar_dir: str):
    tar_filenames = expand_files(tar_dir, file_pattern='*.tar.bz2')
    for tar_filename in tar_filenames:
        with tarfile.open(tar_filename, 'r:bz2') as tar:
            members = tar.getmembers()
            logger.info(f'{tar_filename} has {len(members)} entries')
            for member in tar.getmembers():
                if not member.name.endswith('.json'):
                    logger.info(f'skipping {member.name}')
                    continue
                f = tar.extractfile(member)
                if f is None:
                    logger.warning(f'None from extracting {member}')
                    continue
                content = f.read()
                # content is bytes, so we should decode right? I think json.loads does automatically
                # if hasattr(content, 'decode'):
                #     content = content.decode('utf-8')
                jobj = json.loads(content)
                jobj['filename'] = os.path.split(tar_filename)[-1] + '/' + member.name
                yield jobj


if __name__ == '__main__':
    logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    opts = Options()
    fill_from_args(opts)
    domain_counts = Counter()
    style_counts = Counter()
    filter_reasons = Counter()
    report = Reporting()
    total_count = 0
    wrote_count = 0
    with shuffled_writer(opts.output) as out:
        for jobj in json_from_tars(opts.input):
            if report.is_time():
                report.display()
                logger.info(f'wrote {wrote_count} tables after filtering')
            total_count += 1
            of_interest = True
            cols = jobj['relation']
            if len(cols) < opts.min_cols:
                style_counts['few columns'] += 1
                of_interest = False
            assert all(len(c) == len(cols[0]) for c in cols)
            if len(cols[0]) < opts.min_rows:
                style_counts['few rows'] += 1
                of_interest = False
            if not jobj['hasHeader']:
                style_counts['no header'] += 1
                of_interest = False
            url = jobj['url']
            domain_start = url.find('//')
            if domain_start == -1:
                style_counts['bad domain'] += 1
                if style_counts['bad domain'] < 10:
                    print(f'url = {url}')
                of_interest = False
            domain_end = url.find('/', domain_start + 2)
            if domain_end == -1:
                domain_end = len(url)
            domain = url[domain_start + 2:domain_end]
            if 'wikipedia' in domain:  # or 'wikipedia' in jobj['pageTitle'].lower():
                style_counts['wikipedia'] += 1
                of_interest = False

            if not of_interest:
                continue
            assert 0 <= jobj['headerRowIndex'] < len(jobj['relation'][0])
            style_counts['of interest'] += 1
            domain_counts[domain] += 1
            header_position = jobj['headerPosition']
            header_row_index = jobj['headerRowIndex']
            style_counts[f'header:{header_position},{header_row_index}'] += 1
            table_orientation = jobj['tableOrientation']
            style_counts[f'orientation:{table_orientation}'] += 1
            page_title = jobj['pageTitle']
            title = jobj['title']
            has_key_column = jobj['hasKeyColumn']
            key_column_index = jobj['keyColumnIndex']
            style_counts[f'key column:{has_key_column},{key_column_index}'] += 1
            if key_column_index >= len(jobj['relation']):
                key_column_index = -1
                style_counts['bad key column'] += 1
            # jobj['textAfterTable']
            # jobj['textBeforeTable']  # maybe keep some of this? Like the last 100 chars, broken on a word boundary
            tbl = Table.from_webtable(jobj)
            clean(tbl)
            filter_reason = filter(tbl)
            if filter_reason is None:
                out.write(json.dumps(tbl.to_dict())+'\n')
                wrote_count += 1
            else:
                filter_reasons[filter_reason] += 1
    with write_open(opts.report) as out:
        out.write(f'Total: {total_count}\n')
        for style, count in style_counts.items():
            out.write(f'{style}: {count}\n')
        out.write('\n\n')
        domain_counts_list = [(domain, count) for domain, count in domain_counts.items() if count >= 10]
        domain_counts_list.sort(key=lambda x:x[1], reverse=True)
        out.write('Most Frequent Domains\n=============================\n')
        for domain, count in domain_counts_list[:100]:
            out.write(f'{domain}: {count}\n')
        out.write('\n\n')
        out.write('Filter Reasons\n=============================\n')
        filter_reason_list = [(reason, count) for reason, count in filter_reasons.items()]
        filter_reason_list.sort(key=lambda x:x[1], reverse=True)
        for reason, count in filter_reason_list:
            out.write(f'{reason}: {count}\n')
