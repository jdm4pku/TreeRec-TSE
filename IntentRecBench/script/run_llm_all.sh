#!/bin/bash
# ========================================================
# Run LLM recommendation for all ecosystems and models
# Author: Auto-generated
# ========================================================

# ---- 全局配置 ----
DATA_DIR="IntentRecBench/data"
OUTPUT_ROOT="output/baselines"
TOP_K=5
P_K="1 2 3 4"
DCG_K="2 3 4 5"

# 两阶段策略配置
USE_TWO_STAGE="--use_two_stage"   # 使用两阶段策略（默认启用，使用 --no_two_stage 禁用）
FILTER_TOP_PERCENT=0.1            # 第一阶段筛选出的候选制品百分比（默认0.1，即top 10%）

# ---- 定义要跑的生态系统 ----
ECOSYSTEMS=("linux") # "hf" "js" 

# ---- 定义要跑的模型 ----
MODELS=(
    # "gpt-4o"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "Qwen/Qwen3-32B"
    "deepseek-ai/DeepSeek-R1"
    # "meta-llama/llama-3.1-8b-instruct"
)

# ---- 主循环 ----
for MODEL_NAME in "${MODELS[@]}"; do
    # 模型别名（去掉特殊字符，方便做目录名）
    MODEL_ALIAS=$(echo "$MODEL_NAME" | tr '/-' '__')

    # 模型独立输出目录
    OUTPUT_DIR="${OUTPUT_ROOT}/LLM/${MODEL_ALIAS}"
    mkdir -p "$OUTPUT_DIR"

    echo "============================================================="
    echo "🧠 Running LLM model: ${MODEL_NAME}"
    echo "   Output directory: ${OUTPUT_DIR}"
    echo "============================================================="

    for ECO in "${ECOSYSTEMS[@]}"; do
        echo "----------------------------------------------"
        echo "🚀 Running recommendation for ecosystem: ${ECO}"
        echo "----------------------------------------------"

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

        # 检查执行状态
        if [ $? -ne 0 ]; then
            echo "❌ ${MODEL_NAME} failed on ${ECO}"
        else
            echo "✅ ${MODEL_NAME} finished on ${ECO}"
        fi

        echo "----------------------------------------------"
        sleep 2  # 防止API限流
    done

    echo ""
    echo "🏁 Model ${MODEL_NAME} completed for all ecosystems."
    echo ""
done

echo "============================================================="
echo "🎉 All models and ecosystems completed!"
echo "Results saved under: ${OUTPUT_ROOT}/LLM/"
echo "============================================================="