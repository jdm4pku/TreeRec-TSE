#!/bin/bash
# ========================================================
# Run LLM recommendation for all ecosystems
# Author: Auto-generated
# ========================================================

# ---- 全局配置 ----
DATA_DIR="IntentRecBench/data"
OUTPUT_DIR="output/baselines"
# 支持的模型示例:
# GPT: gpt-4o, gpt-4-turbo, gpt-3.5-turbo
# Qwen: Qwen/Qwen3-14B, Qwen/Qwen3-8B, Qwen/Qwen3-32B
# DeepSeek: Pro/deepseek-ai/DeepSeek-R1, deepseek-ai/DeepSeek-V3
# Llama: meta-llama/llama-3.1-8b-instruct, meta-llama/llama-3.1-70b-instruct
MODEL_NAME="Qwen/Qwen3-14B"  # 可以根据需要修改模型名称
TOP_K=5
P_K="1 2 3 4"
DCG_K="2 3 4 5"
# 两阶段策略配置
USE_TWO_STAGE="--use_two_stage"  # 使用两阶段策略（默认启用，使用 --no_two_stage 禁用）
FILTER_TOP_PERCENT=0.1  # 第一阶段筛选出的候选制品百分比（默认0.1，即top 10%）
SCORING_BATCH_SIZE=100 # linux：20, hf:30, js:100

# 批量打分的大小（默认20，即每次批量打分20个制品）
# ---- 定义要跑的生态系统 ----
ECOSYSTEMS=("hf") # "hf" "js" "linux"

# ---- 主循环 ----
for ECO in "${ECOSYSTEMS[@]}"; do
    echo "=============================================="
    echo "🚀 Running LLM recommendation for ecosystem: ${ECO}"
    echo "   Model: ${MODEL_NAME}"
    echo "=============================================="

    python IntentRecBench/src/baselines/llm.py \
        --data_dir "$DATA_DIR" \
        --ecosystem "$ECO" \
        --output_dir "$OUTPUT_DIR" \
        --model_name "$MODEL_NAME" \
        --top_k $TOP_K \
        --p_k $P_K \
        --dcg_k $DCG_K \
        $USE_TWO_STAGE \
        --filter_top_percent $FILTER_TOP_PERCENT

    echo "✅ ${ECO} finished."
    echo "----------------------------------------------"
    sleep 2  # 防止日志文件时间戳冲突
done

echo "=============================================="
echo "🎉 All ecosystems completed!"
echo "=============================================="

