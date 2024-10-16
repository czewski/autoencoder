#!/bin/bash

#
COM FOLD E SEM FOLD
TEST MULTIHEAD?
#

# Datasets to apply
datasets=("data/diginetica/" "data/yoochoose1_4/" "data/yoochoose1_64/")

# Experiment 1: Base hyperparameters test
# for dataset in "${datasets[@]}"; do
#   for batch_size in 64 128 256; do
#     for hidden_size in 100 150 200 250; do
#       for embed_dim in 50 100 150 200; do
#         for lr in 0.001 0.0001 0.00001; do
#           for weight_decay in 0.00001 0.0001; do
#             python main.py \
#               --dataset_path $dataset \
#               --batch_size $batch_size \
#               --hidden_size $hidden_size \
#               --embed_dim $embed_dim \
#               --lr $lr \
#               --weight_decay $weight_decay \
#               --lr_dc 0.1 \
#               --lr_dc_step 40 \
#               --epoch 50 \
#               --topk 20 \
#               --valid_portion 0.1 \
#               --max_len 15 \
#               --alignment_function "sdp" \
#               --pos_enc True \
#               --knn False \
#               --embeddings "random" \
#               --folds 5
#           done
#         done
#       done
#     done
#   done
# done

# Actual experiment 1:
for dataset in "${datasets[@]}"; do
  for batch_size in 64 128; do
    for hidden_size in 100 150 200; do
      for embed_dim in 50 100 150; do
        for lr in 0.001 0.0001 0.00001; do
          for weight_decay in 0.0001; do
            python main.py \
              --dataset_path $dataset \
              --batch_size $batch_size \
              --hidden_size $hidden_size \
              --embed_dim $embed_dim \
              --lr $lr \
              --weight_decay $weight_decay \
              --lr_dc 0.1 \
              --lr_dc_step 40 \
              --epoch 50 \
              --topk 20 \
              --valid_portion 0.1 \
              --max_len 15 \
              --alignment_function "sdp" \
              --pos_enc True \
              --knn False \
              --embeddings "random" \
              --folds 5
          done
        done
      done
    done
  done
done

# Experiment 2: Attention types + pos_enc (Best from 1 assumed as placeholders)
best_batch_size=64
best_hidden_size=150
best_embed_dim=50
best_lr=0.001
best_weight_decay=0.00001
for dataset in "${datasets[@]}"; do
  for alignment_function in "sdp" "dp" "additive" "concat" "biased_general" "general"; do
    python main.py \
      --dataset_path $dataset \
      --batch_size $best_batch_size \
      --hidden_size $best_hidden_size \
      --embed_dim $best_embed_dim \
      --lr $best_lr \
      --weight_decay $best_weight_decay \
      --lr_dc 0.1 \
      --lr_dc_step 40 \
      --epoch 50 \
      --topk 20 \
      --valid_portion 0.1 \
      --max_len 15 \
      --alignment_function $alignment_function \
      --pos_enc True \
      --knn False \
      --embeddings "random" \
      --folds 5
  done
done

# Experiment 3: Validate use of pos_enc, LSTM, both (Best parameters from 1 and 2)
best_alignment_function="sdp"
for dataset in "${datasets[@]}"; do
  for pos_enc in "True" "False" "Both"; do
    python main.py \
      --dataset_path $dataset \
      --batch_size $best_batch_size \
      --hidden_size $best_hidden_size \
      --embed_dim $best_embed_dim \
      --lr $best_lr \
      --weight_decay $best_weight_decay \
      --lr_dc 0.1 \
      --lr_dc_step 40 \
      --epoch 50 \
      --topk 20 \
      --valid_portion 0.1 \
      --max_len 15 \
      --alignment_function $best_alignment_function \
      --pos_enc $pos_enc \
      --knn False \
      --embeddings "random" \
      --folds 5
  done
done

# Experiment 4: Validate embeddings type
for dataset in "${datasets[@]}"; do
  for embeddings in "random" "item2vec" "glove"; do
    python main.py \
      --dataset_path $dataset \
      --batch_size $best_batch_size \
      --hidden_size $best_hidden_size \
      --embed_dim $best_embed_dim \
      --lr $best_lr \
      --weight_decay $best_weight_decay \
      --lr_dc 0.1 \
      --lr_dc_step 40 \
      --epoch 50 \
      --topk 20 \
      --valid_portion 0.1 \
      --max_len 15 \
      --alignment_function $best_alignment_function \
      --pos_enc "True" \
      --knn "False" \
      --embeddings $embeddings \
      --folds 5
  done
done

# Experiment 5: Validate knn
best_embeddings="item2vec"  # Assuming best embeddings from 4
for dataset in "${datasets[@]}"; do
  for knn in True False; do
    python main.py \
      --dataset_path $dataset \
      --batch_size $best_batch_size \
      --hidden_size $best_hidden_size \
      --embed_dim $best_embed_dim \
      --lr $best_lr \
      --weight_decay $best_weight_decay \
      --lr_dc 0.1 \
      --lr_dc_step 40 \
      --epoch 50 \
      --topk 20 \
      --valid_portion 0.1 \
      --max_len 15 \
      --alignment_function $best_alignment_function \
      --pos_enc True \
      --knn $knn \
      --embeddings $best_embeddings \
      --folds 5
  done
done

# Experiment 6.1: Validate max_len
for dataset in "${datasets[@]}"; do
  for max_len in 20 19 18 17 16 15 14 13 12 11 10; do
    python main.py \
      --dataset_path $dataset \
      --batch_size $best_batch_size \
      --hidden_size $best_hidden_size \
      --embed_dim $best_embed_dim \
      --lr $best_lr \
      --weight_decay $best_weight_decay \
      --lr_dc 0.1 \
      --lr_dc_step 40 \
      --epoch 50 \
      --topk 20 \
      --valid_portion 0.1 \
      --max_len 15 \
      --alignment_function $best_alignment_function \
      --pos_enc True \
      --knn $knn \
      --embeddings $best_embeddings \
      --folds 5
  done
done

# Experiment 6.2: Validate max_len
for dataset in "${datasets[@]}"; do
  for max_len in 9 8 7 6 5 4; do
    python main.py \
      --dataset_path $dataset \
      --batch_size $best_batch_size \
      --hidden_size $best_hidden_size \
      --embed_dim $best_embed_dim \
      --lr $best_lr \
      --weight_decay $best_weight_decay \
      --lr_dc 0.1 \
      --lr_dc_step 40 \
      --epoch 50 \
      --topk 20 \
      --valid_portion 0.1 \
      --max_len 15 \
      --alignment_function $best_alignment_function \
      --pos_enc True \
      --knn $knn \
      --embeddings $best_embeddings \
      --folds 5
  done
done

# Experiment 7: Validate output_operation (also, i added relu to "mean" (which was default), se if changes anything)

# Experiment 8: Validate multi_head

