from torch_util.hypers_base import HypersBase
from corpus.corpus_client import CorpusClient
from transformers import RagTokenizer, RagSequenceForGeneration, RagTokenForGeneration
from util.line_corpus import read_lines
import logging
from table_augmentation.augmentation_tasks import TaskOptions
from table_augmentation.table import Table
import ujson as json

logger = logging.getLogger(__name__)


class RagHypers(HypersBase):
    def __init__(self):
        super().__init__()
        self.tables = ''
        self.task = TaskOptions()
        self.is_query = False  # set flag true if the 'tables' we are reading already are queries
        self.model_name = 'facebook/rag-token-nq'
        self.model_path = ''
        self.corpus_endpoint = ''
        self.port = 5001  # for starting our own corpus server
        self.n_docs = 5
        self.max_context_length = 512
        self.max_target_length = 512
        # only used for train
        self.num_instances = -1
        self.warmup_fraction = 0.1
        self.__required_args__ = ['tables', 'output_dir', 'corpus_endpoint']

    def _post_init(self):
        super()._post_init()
        if self.num_instances == -1:
            if self.is_query:
                self.num_instances = sum(1 for _ in read_lines(self.tables))
            else:
                query_maker = self.task.get_query_maker()
                self.num_instances = sum(1 for line in read_lines(self.tables)
                                         for _ in query_maker(Table.from_dict(json.loads(line))))
            logger.info(f'Counted num_instances = {self.num_instances}')

    def cleanup_corpus_server(self):
        CorpusClient.cleanup_corpus_server(self)

    def get_tokenizer_and_model(self):
        # initialize the model and index
        tokenizer = RagTokenizer.from_pretrained(self.model_name)
        # rag_conf = RagConfig.from_pretrained(opts.model_name)
        if 'rag-token' in self.model_name:
            model = RagTokenForGeneration.from_pretrained(self.model_path if self.model_path else self.model_name)
        elif 'rag-sequence' in self.model_name:
            model = RagSequenceForGeneration.from_pretrained(self.model_path if self.model_path else self.model_name)
        else:
            raise AssertionError
        return tokenizer, model
