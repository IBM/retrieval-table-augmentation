FAIRSEQ_CODE=/data/fairseq
FAIRSEQ_DATA=/data/TableAugmentation/queries/fairseq
arch=base

for SPLIT in train dev test
do
  for LANG in inp oup
  do
    python ${FAIRSEQ_CODE}/examples/roberta/multiprocessing_bpe_encoder.py \
      --encoder-json ${FAIRSEQ_DATA}/encoder.json \
      --vocab-bpe ${FAIRSEQ_DATA}/vocab.bpe \
      --inputs "$SPLIT.$LANG" \
      --outputs "$SPLIT.bpe.$LANG" \
      --workers 60 \
      --keep-empty;
  done
done

fairseq-preprocess \
  --source-lang inp \
  --target-lang oup \
  --trainpref "train.bpe" \
  --validpref "dev.bpe" \
  --destdir "bin-${arch}/" \
  --workers 60 \
  --srcdict ${FAIRSEQ_DATA}/bart.${arch}/dict.txt \
  --tgtdict ${FAIRSEQ_DATA}/bart.${arch}/dict.txt