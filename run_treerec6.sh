#!/bin/bash
# ========================================================
# Run TreeRec recommendation for all ecosystems (with rerank)
# Author: Dongming Jin
# ========================================================

# ---- 全局配置 ----
DATA_DIR="IntentRecBench/data"
OUTPUT_DIR="output"
LLM_NAME="gpt4o"

RERANK_MODEL="gpt-4o-2024-05-13"
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
SUMMARIZATION_MODEL="gpt-4o-2024-05-13"
TREE_BUILDER_TYPE="cluster"

TOP_K=20
P_K="1 2 3 4"
DCG_K="2 3 4 5"

# 这里改成你刚才那段 Python 文件的真实路径
SCRIPT_PATH="run_treerec.py"

# ---- 定义要跑的生态系统 ----
ECOSYSTEMS=("hf") # "linux" "js" "hf"

# ---- 主循环 ----
for ECO in "${ECOSYSTEMS[@]}"; do
    echo "=============================================="
    echo "🌲 Running TreeRec recommendation for ecosystem: ${ECO}"
    echo "=============================================="

    # 注意：此脚本使用 rerank（添加 --use_rerank 参数）
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT_PATH" \
        --data_dir "$DATA_DIR" \
        --llm_name "$LLM_NAME" \
        --ecosystem "$ECO" \
        --output_dir "$OUTPUT_DIR" \
        --rerank_model "$RERANK_MODEL" \
        --embedding_model "$EMBEDDING_MODEL" \
        --summarization_model "$SUMMARIZATION_MODEL" \
        --tree_builder_type "$TREE_BUILDER_TYPE" \
        --tr_threshold 0.5 \
        --tr_top_k 40 \
        --tr_selection_mode "top_k" \
        --tb_max_tokens 100 \
        --tb_num_layers 5 \
        --tb_threshold 0.5 \
        --tb_top_k 5 \
        --tb_selection_mode "top_k" \
        --tb_summarization_length 100 \
        --top_k "$TOP_K" \
        --p_k $P_K \
        --dcg_k $DCG_K
    # 添加了 --use_rerank 参数，因此会使用 rerank 模型

    echo "✅ ${ECO} finished."
    echo "----------------------------------------------"
    sleep 2  # 防止日志文件时间戳冲突
done