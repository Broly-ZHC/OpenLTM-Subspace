#!/usr/bin/env bash

python run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model timer_xl_subspace \
  --data MultivariateDatasetBenchmark \
  --seq_len 96 \
  --input_token_len 96 \
  --output_token_len 96 \
  --test_seq_len 96 \
  --test_pred_len 96 \
  --batch_size 16
