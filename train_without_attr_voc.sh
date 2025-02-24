export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:/data/lihl/fsod

/root/miniconda3/envs/fsod/bin/python3  train/train_without_attr_voc.py \
    --model-name 'llama3.1-8b' \
    --processed-voc-data '/data/lihl/LLaFS2/data/sft_data_without_attr/all_class_dataset_with_description.jsonl' \
    --data-path '/data3/lihl/sft_dataset_20250115' \
    --ft-mode 'lora' \
    --prepare-dataset 0 \
    --lora-rank 32 \
    --lr 2e-4 \
    --epoch 50 \
    --max-seq-length 1024

