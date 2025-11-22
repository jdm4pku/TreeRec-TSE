#!/bin/bash
# ========================================================
# Run GPT-5 evaluation for dataset reliability evaluation
# 使用GPT-5评估数据集样本可靠性，并计算与GPT-4o评估结果的一致率
# Author: Auto-generated
# ========================================================

# ---- 全局配置 ----
DATA_DIR="IntentRecBench/data"
GPT4O_OUTPUT_DIR="IntentRecBench/data/evaluation_results"
GPT5_OUTPUT_DIR="IntentRecBench/data/gpt5_evaluation_results"
# GPT-5模型配置（可以通过环境变量覆盖）
GPT5_MODEL="${GPT5_MODEL:-gpt-5}"  # 默认使用gpt-5
# 采样比例（10%）
SAMPLE_RATIO=0.1
# 随机种子（确保可复现）
SEED=42

# ---- 检查Python环境 ----
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到python3命令"
    exit 1
fi

# ---- 检查数据目录 ----
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ 错误: 数据目录不存在: $DATA_DIR"
    exit 1
fi

# ---- 检查GPT-4o结果目录 ----
if [ ! -d "$GPT4O_OUTPUT_DIR" ]; then
    echo "⚠️  警告: GPT-4o评估结果目录不存在: $GPT4O_OUTPUT_DIR"
    echo "   将无法计算与GPT-4o的一致率"
fi

# ---- 创建输出目录 ----
mkdir -p "$GPT5_OUTPUT_DIR"

# ---- 显示配置信息 ----
echo "=============================================="
echo "📊 GPT-5数据集可靠性评估"
echo "=============================================="
echo "数据目录: $DATA_DIR"
echo "GPT-4o结果目录: $GPT4O_OUTPUT_DIR"
echo "GPT-5结果目录: $GPT5_OUTPUT_DIR"
echo "使用模型: $GPT5_MODEL"
echo "采样比例: $(echo "$SAMPLE_RATIO * 100" | bc -l)%"
echo "随机种子: $SEED"
echo "说明: 每个生态将采样10%的样本进行评估"
echo "=============================================="
echo ""

# ---- 设置环境变量 ----
export GPT5_MODEL="$GPT5_MODEL"
# 如果需要，可以在这里设置API密钥和base_url
# export OPENAI_API_KEY="your-api-key"
# export OPENAI_BASE_URL="your-base-url"

# ---- 运行评估 ----
python3 IntentRecBench/src/data_construction/gpt5_evaluation.py \
    --data_dir "$DATA_DIR" \
    --gpt4o_output_dir "$GPT4O_OUTPUT_DIR" \
    --gpt5_output_dir "$GPT5_OUTPUT_DIR" \
    --model "$GPT5_MODEL" \
    --sample_ratio $SAMPLE_RATIO \
    --seed $SEED

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "🎉 GPT-5评估完成！"
    echo "=============================================="
    echo "评估结果保存在: $GPT5_OUTPUT_DIR"
    echo "  - 详细结果: ${GPT5_OUTPUT_DIR}/{ecosystem}_gpt5_evaluation_results.json"
    echo "  - 汇总统计: ${GPT5_OUTPUT_DIR}/gpt5_evaluation_summary.json"
    echo "=============================================="
else
    echo ""
    echo "❌ GPT-5评估失败"
    exit 1
fi

