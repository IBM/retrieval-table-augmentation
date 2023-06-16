from util.line_corpus import write_open
from util.args_help import fill_from_args
import ujson as json
import os


class Options:
    def __init__(self):
        self.table_dir = ''
        self.split_definitions = ''


def main():
    opts = Options()
    fill_from_args(opts)
    test_tables = set()
    for file in os.listdir(opts.split_definitions):
        with open(os.path.join(opts.split_definitions, file)) as f:
            test_table_list = json.load(f)
            for table_id in test_table_list:
                test_tables.add(table_id)
    low_row_count = 0
    low_col_count = 0
    repeated_header_count = 0
    low_row_count_train = 0
    low_col_count_train = 0
    repeated_header_count_train = 0
    for file in os.listdir(opts.table_dir):
        with open(os.path.join(opts.table_dir, file)) as f:
            tables = json.load(f)
        for table_id, table in tables.items():
            header = table['title']
            title = table['pgTitle'] + '\t' + table['caption']
            rows = table['data']
            assert all([len(row) == len(header) for row in rows])
            if table_id in test_table_list:
                if len(header) < 4:
                    low_col_count += 1
                if len(rows) < 4:
                    low_row_count += 1
                if any([h == header[0] for h in header[1:]]):
                    repeated_header_count += 1
            else:
                if len(header) < 4:
                    low_col_count_train += 1
                if len(rows) < 4:
                    low_row_count_train += 1
                if any([h == header[0] for h in header[1:]]):
                    repeated_header_count_train += 1
    print('In Test:')
    print(f'low row count = {low_row_count}')
    print(f'low col count = {low_col_count}')
    print(f'repeated header count = {repeated_header_count}')
    print('In Train:')
    print(f'low row count = {low_row_count_train}')
    print(f'low col count = {low_col_count_train}')
    print(f'repeated header count = {repeated_header_count_train}')


if __name__ == "__main__":
    main()