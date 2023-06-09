#! /bin/bash
set -e
work_dir=path_to_your_project
code_dir=$work_dir/fairseq-kd
data_dir=path_to_your_data
bin_dir=$data_dir/data-bin

export PYTHONPATH=$code_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

setting=base_wmt14ende_vanilla_word_kd
output_dir=$work_dir/ckpts/$setting

if [ ! -d $output_dir ];then
    mkdir -p $output_dir
fi

slang=en
tlang=de

python -u $code_dir/fairseq_cli/train.py $bin_dir \
    --task translation_kd --arch transformer \
    --teacher-arch transformer_vaswani_wmt_en_de_big \
    --teacher-checkpoint ${work_dir}/ckpts/big_wmt14ende_teacher/checkpoint_best.pt \
    --kd-rate 0.5 \
    --share-all-embeddings \
    --source-lang $slang --target-lang $tlang \
    --dropout 0.1 --weight-decay 0.0 --criterion label_smoothed_cross_entropy_kd \
    --attention-dropout 0.1 --activation-dropout 0.1 \
    --label-smoothing 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 5.0 \
    --lr 7e-4 --lr-scheduler inverse_sqrt \
    --valid-subset valid,test \
    --eval-bleu --eval-bleu-args '{"beam": 4, "lenpen": 0.6}' \
    --eval-tokenized-bleu --eval-bleu-remove-bpe \
    --validate-interval-updates 5000 --validate-interval 9999999 \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --max-tokens 4096 --update-freq 2 --max-update 200000 \
    --save-interval-updates 5000 --save-interval 9999999 \
    --keep-interval-updates 50 --keep-best-checkpoints 5 \
    --distributed-world-size 4 --ddp-backend=legacy_ddp --log-interval 100 \
    --num-workers 8 \
    --save-dir $output_dir \
    --fp16 \
    > $output_dir/train.log 2>&1 &


