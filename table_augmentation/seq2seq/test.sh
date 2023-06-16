ckpt=$1
data_dir=$2
B=${3:-"35"}

fairseq-interactive ${data_dir}/bin-base --path $ckpt --beam $B --nbest $B --remove-bpe --buffer-size 1024 --max-tokens 4096 --fp16 > $ckpt.beam-$B.out < ${data_dir}/test.bpe.inp
bash $( dirname $0 )/eval/convert_fairseq_output_to_text.sh $ckpt.beam-$B.out

# TODO: answer normalization option
# python ${PYTHONPATH}/table_augmentation/seq2seq/eval_fairseq.py \
# --text_out $ckpt.beam-$B.out.text \
# --target_text ${data_dir}/test.oup --answer_normalization deunicode --beam_size $B
