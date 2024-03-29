from transformers import (RagTokenForGeneration, DPRQuestionEncoder)
from util.args_help import fill_from_args


class Options:
    def __init__(self):
        self.rag_model_name = 'facebook/rag-token-nq'
        self.qry_encoder_path = 'facebook/dpr-question_encoder-multiset-base'
        self.save_dir = ''
        self.__required_args__ = ['save_dir']


opts = Options()
fill_from_args(opts)

rag_model = RagTokenForGeneration.from_pretrained(opts.rag_model_name)

qencoder = DPRQuestionEncoder.from_pretrained(opts.qry_encoder_path)

rag_qenocder = rag_model.question_encoder
rag_qenocder.load_state_dict(qencoder.state_dict(), strict=True)

rag_model.save_pretrained(opts.save_dir)
