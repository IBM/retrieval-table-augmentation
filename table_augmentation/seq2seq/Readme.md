```bash
conda create --name fairseq python=3.7
conda activate fairseq
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
pip install ujson
```

```bash
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz
tar -xvf bart.base.tar.gz
```