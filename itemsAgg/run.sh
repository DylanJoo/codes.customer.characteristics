CUDA_LAUNCH_BLOCKING=1

python3 train.py \
  --model_name_or_path bert-base-chinese \
  --config_name ckiplab/bert-base-chinese \
  --tokenizer_name ckiplab/bert-base-chinese \
  --output_dir checkpoints/itemsAgg.v1 \
  --max_length 20 \
  --per_device_train_batch_size 128 \
  --max_steps 5000 \
  --save_steps 1000 \
  --do_train
