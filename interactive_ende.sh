#!/bin/bash
set -e
work_dir=path_to_your_project
code_dir=$work_dir/fairseq-kd
data_dir=path_to_your_data
bin_dir=$data_dir/data-bin
moses_dir=path_to_mosesdecoder

export PYTHONPATH=$code_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

setting=base_wmt14ende_student_baseline
output_dir=$work_dir/ckpts/$setting
result_dir=$work_dir/results/$setting

if [ ! -d $output_dir ];then
    mkdir -p $output_dir
fi

if [ ! -d $result_dir ];then
    mkdir -p $result_dir
fi

slang=en
tlang=de

subset=test

python -u $code_dir/fairseq_cli/interactive.py $bin_dir \
    --task translation \
    --input ${data_dir}/${subset}.en \
    --source-lang $slang \
    --target-lang $tlang \
    --path ${output_dir}/checkpoint_last5avg_20w.pt \
    --beam 4 --lenpen 0.6 \
    --batch-size 128 \
    --buffer-size 256 \
    > ${result_dir}/${subset}.hyp.ori

grep ^H ${result_dir}/${subset}.hyp.ori | cut -f3- > ${result_dir}/${subset}.hyp.bpe
sed -r 's/(@@ )| (@@ ?$)//g' < ${result_dir}/${subset}.hyp.bpe > ${result_dir}/${subset}.hyp.tok
sed -r 's/(@@ )| (@@ ?$)//g' < ${data_dir}/${subset}.de > ${result_dir}/${subset}.ref.tok

perl ${moses_dir}/scripts/generic/multi-bleu.perl ${result_dir}/${subset}.ref.tok < ${result_dir}/${subset}.hyp.tok


