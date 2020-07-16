export CUDA_VISIBLE_DEVICES=0
python run_ner.py
    --data_dir ./data/resume/ \
    --model_type bert \
    --model_name_or_path bert-base-chinese \
    --output_dir ./pretrain_bert \
    --cache_dir /data/project/TEMP_transformers \
    --max_seq_length 256 \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 20 \
    --logging_steps 200 



