import requests
import numpy as np
import json
import logging
import time
from util.args_help import fill_from_args
from table_augmentation.augmentation_tasks import Table
from util.reporting import Reporting
from typing import List, Dict
from util.line_corpus import jsonl_lines, block_shuffle

logger = logging.getLogger(__name__)


class TableRetrievalClient:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint  # 'http://localhost:5001'
        self.retrieval_time = 0
        self.headers = {'Content-Type': 'application/json'}
        # get config info from server
        config = requests.get(self.endpoint+'/config', headers=self.headers).json()
        print(config)
        self.reporting = Reporting()
        self.rest_dtype = np.float32 if config['dtype'] == 32 else np.float16

    def retrieve(self, tables: List[Table], k: int) -> Dict[str, List[Dict[str, List[str]]]]:
        """
        :param tables: the tables to retrieve related tables for
        :param k: number of table parts to retrieve per query
        :return: docs: map from 'col' and 'row' to list of dict [{title: [t_1...t_k], text: [...], pid: [...]}] * batch_size
        """
        query = {'tables': [t.to_dict() for t in tables], 'k': k}
        start_time = time.time()
        response = requests.post(self.endpoint+'/retrieve', data=json.dumps(query), headers=self.headers)
        self.retrieval_time += time.time() - start_time
        try:
            rdocs = response.json()
        except json.decoder.JSONDecodeError:
            logger.error(f'Bad response: {response}\nRequest: {json.dumps(query)}')
            raise json.decoder.JSONDecodeError

        return rdocs


class Options:
    def __init__(self):
        self.endpoint = ''
        self.tables = ''
        self.k = 5


if __name__ == '__main__':
    opts = Options()
    fill_from_args(opts)
    client = TableRetrievalClient(opts.endpoint)

    # retrieve a table
    """
    test_table = {'table_id': 'my special table', 'title': 'Employees', 
                  'header': ['Employee ID', 'Employee Name', 'Department'],
                  'rows': [['0', 'Michael', 'AI'], ['1', 'Alfio', 'AI'], ['2', 'Faisal', 'AI']],
                  'key_col_ndx': 1}
    """
    test_table = {'table_id': 'my special table', 'title': 'Malware',
                  'header': ['Name', 'Target', 'Year'],
                  'rows': [['WannaCry', 'Microsoft', '2017'], ['Poison Ivy', 'Microsoft', '2005']]}

    rdocs = client.retrieve([Table.from_dict(test_table)], opts.k)
    print(test_table)
    print('=' * 80)
    print(rdocs)
    k = input('...')

    for line in block_shuffle(jsonl_lines(opts.tables)):
        jobj = json.loads(line)
        table = Table.from_dict(jobj)
        rdocs = client.retrieve([table], opts.k)
        print(jobj)
        print('=' * 80)
        print(rdocs)
        k = input('...')