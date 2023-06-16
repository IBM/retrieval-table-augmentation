from util.line_corpus import write_open
from util.corpus_reader import corpus_reader
import ujson as json
from dpr.retriever_bm25 import RetrieverBM25
from util.args_help import fill_from_args
import os
from typing import Union, Optional


class Options:
    def __init__(self):
        self.input = ''
        self.output_dir = ''
        self.file_count = 8
        self.jar = ''
        self.replace_title_text_sep_in_title = False
        self._required_args = ['input', 'output_dir']


def anserini_convert(input: Union[str, bytes, os.PathLike], output_dir: str, *,
                     file_count: int = 8,
                     title_text_sep: str = RetrieverBM25.TITLE_TEXT_SEP,
                     replace_title_text_sep_in_title: bool = False):
    """
    Convert corpus with pid/title/text into id/context for Anserini
    :param input: input file or directory
    :param output_dir:
    :param file_count: number of files to create
    :param title_text_sep: separator between title and text in contents field
    :param replace_title_text_sep_in_title: if set, we will replace the title_text_sep with space if it occurs in the title
    :return:
    """
    #
    outfiles = [write_open(os.path.join(output_dir, 'tmp', f'{j}.json')) for j in range(file_count)]
    pids = set()
    for line_ndx, passage in enumerate(corpus_reader(input)):
        f = outfiles[line_ndx % len(outfiles)]
        if passage.pid in pids:
            raise ValueError(f'Duplicate pid: {passage.pid}')
        pids.add(passage.pid)
        title = passage.title.replace(title_text_sep, ' ') if replace_title_text_sep_in_title else passage.title
        anserini_passage = {'id': passage.pid,
                            'contents': title+title_text_sep+passage.text}
        if not replace_title_text_sep_in_title:
            anserini_passage['title_text_sep'] = [len(passage.title), len(title_text_sep)]
        f.write(json.dumps(anserini_passage)+'\n')
    for of in outfiles:
        of.close()


def pyserini_index(output_dir: str):
    os.system(f'python -m pyserini.index.lucene --collection JsonCollection \
    --input {os.path.join(output_dir, "tmp")} --generator DefaultLuceneDocumentGenerator \
    --threads 32 --storePositions --storeDocvectors --storeRaw --index {os.path.join(output_dir, "index")}')
    import shutil
    shutil.rmtree(os.path.join(output_dir, 'tmp'))


def anserini_index(output_dir: str, jar: Optional[str]):
    import jnius_config
    jnius_config.set_classpath(jar)
    from jnius import autoclass
    args = [
        '-input', os.path.join(output_dir, 'tmp'),
        '-index', os.path.join(output_dir, 'index'),
        '-collection', 'JsonCollection',
        '-threads', '32', '-storePositions', '-storeDocvectors',
        # NOTE: depends on version
        # '-storeRaw',
        '-storeRawDocs',
        # '-generator', 'DefaultLuceneDocumentGenerator',
        '-generator', 'LuceneDocumentGenerator',
    ]
    JIndexCollection = autoclass('io.anserini.index.IndexCollection')
    JIndexCollection.main(args)
    """
    which does:

    sh /data/Anserini/target/appassembler/bin/IndexCollection -collection JsonCollection \
    -generator LuceneDocumentGenerator -threads 32 -input $<opts.output/tmp> \
    -index <opts.output/index> -storePositions -storeDocvectors -storeRawDocs
    """
    import shutil
    shutil.rmtree(os.path.join(output_dir, 'tmp'))


def main(opts: Options):
    anserini_convert(opts.input, opts.output_dir, file_count=opts.file_count,
                     replace_title_text_sep_in_title=opts.replace_title_text_sep_in_title)
    if opts.jar:
        anserini_index(opts.output_dir, opts.jar)
    else:
        pyserini_index(opts.output_dir)


if __name__ == "__main__":
    opts = Options()
    fill_from_args(opts)
    main(opts)
