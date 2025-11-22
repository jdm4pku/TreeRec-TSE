#!/bin/bash
# ========================================================
# Run Modern Embedding (ModelScope or HuggingFace) for all ecosystems
# Author: Jinyu Cai
# ========================================================

# ---- 全局配置 ----
DATA_DIR="IntentRecBench/data"
OUTPUT_DIR="output/baselines"
MODEL_NAME="sentence-transformers/all-mpnet-base-v2"   # 可改成你的 ModelScope 模型名
PROVIDER="sentence"                                           # sentence = ModelScope/HF, openai = OpenAI API
P_K="1 2 3 4"
DCG_K="2 3 4 5"

# ---- 定义要跑的生态系统 ----
ECOSYSTEMS=("hf" "js" "linux")

# ---- 主循环 ----
for ECO in "${ECOSYSTEMS[@]}"; do
    echo "====================================================="
    echo "🚀 Running Modern Embedding for ecosystem: ${ECO}"
    echo "====================================================="

    python IntentRecBench/src/baselines/modern_models.py \
        --data_dir "$DATA_DIR" \
        --ecosystem "$ECO" \
        --output_dir "$OUTPUT_DIR" \
        --model_name "$MODEL_NAME" \
        --provider "$PROVIDER" \
        --p_k $P_K \
        --dcg_k $DCG_K

    echo "✅ ${ECO} finished."
    echo "-----------------------------------------------------"
    sleep 2  # 防止日志文件时间戳冲突
done