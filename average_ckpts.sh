#!/bin/bash
set -e
export PYTHONPATH=$code_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1

work_dir=path_to_your_project
code_dir=$work_dir/fairseq-kd

setting=base_wmt14ende_student_baseline
output_dir=$work_dir/ckpts/$setting

python ${code_dir}/scripts/average_checkpoints.py \
    --inputs ${output_dir} \
    --output ${output_dir}/checkpoint_last5avg_20w.pt \
    --num-update-checkpoints 5 \
    --checkpoint-upper-bound 200000

