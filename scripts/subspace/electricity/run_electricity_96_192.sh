#!/bin/bash

python -u run.py \
    --task_name forecast \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id electricity_96_192_subspace \
    --model timer_xl_subspace \
    --data MultivariateDatasetBenchmark \
    --seq_len 96 \
    --input_token_len 96 \
    --output_token_len 192 \
    --test_seq_len 96 \
    --test_pred_len 192 \
    --batch_size 16 \
    --num_workers 1 \
    --enc_in 321 \
    --num_groups 16 \
    --d_var 64 \
    --vq_beta 0.25
