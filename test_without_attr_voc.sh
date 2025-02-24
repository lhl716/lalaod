#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=$PYTHONPATH:/data/lihl/fsod

# 定义日志文件
LOG_FILE="/data/lihl/fsod/logs/model_test_log_$(date +%Y%m%d_%H%M%S).log"

# 循环遍历 index 为 5131 的倍数，范围从 1 到 20
for i in $(seq 1 20); do
    index=$((5131 * i))
    lora_path="/data3/lihl/llama-lora-output/checkpoint-${index}"
    
    # 运行 Python 脚本并记录日志
    {
        echo "运行开始: checkpoint-${index} $(date)"
        python eval/model_test_v2.py \
            --model-name 'llama3.1-8b' \
            --data-path '/data3/lihl/sft_dataset_20250115' \
            --finetune-mode 'lora' \
            --lora-path "$lora_path" \
            --max-seq-length 1024
        
        # 检查脚本运行结果
        if [ $? -ne 0 ]; then
            echo "运行失败: checkpoint-${index} $(date)"
            exit 1
        fi

        echo "运行成功: checkpoint-${index} $(date)"
    } >> "$LOG_FILE" 2>&1
done

echo "所有任务完成，日志保存在 $LOG_FILE"