#!/bin/bash
# ========================================================
# Run Word2Vec recommendation for all ecosystems
# Author: Jinyu Cai
# ========================================================

# ---- 全局配置 ----
DATA_DIR="IntentRecBench/data"
OUTPUT_DIR="output/baselines"
USE_FIELDS="name type description"
VECTOR_SIZE=200
P_K="1 2 3 4"
DCG_K="2 3 4 5"

# ---- 定义要跑的生态系统 ----
ECOSYSTEMS=("hf" "js" "linux")

# ---- 主循环 ----
for ECO in "${ECOSYSTEMS[@]}"; do
    echo "=============================================="
    echo "🚀 Running Word2Vec recommendation for ecosystem: ${ECO}"
    echo "=============================================="

    python IntentRecBench/src/baselines/Word2Vec.py \
        --data_dir "$DATA_DIR" \
        --ecosystem "$ECO" \
        --output_dir "$OUTPUT_DIR" \
        --use_fields $USE_FIELDS \
        --vector_size $VECTOR_SIZE \
        --p_k $P_K \
        --dcg_k $DCG_K

    echo "✅ ${ECO} finished."
    echo "----------------------------------------------"
    sleep 2  # 防止日志文件时间戳冲突
done