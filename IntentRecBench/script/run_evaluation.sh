#!/bin/bash
# ========================================================
# Run GPT evaluation for dataset reliability evaluation
# Author: Auto-generated
# ========================================================

# ---- 全局配置 ----
DATA_DIR="IntentRecBench/data"
OUTPUT_DIR="IntentRecBench/data/evaluation_results"
# GPT模型配置（可以通过环境变量覆盖）
# 支持的模型: gpt-4o-2024-05-13, gpt-4-turbo, gpt-3.5-turbo, gpt-5 (如果可用)
GPT_MODEL="${GPT_MODEL:gpt-4o-2024-05-13}"  # 默认使用gpt-4o，可通过环境变量改为gpt-5

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

# ---- 创建输出目录 ----
mkdir -p "$OUTPUT_DIR"

# ---- 显示配置信息 ----
echo "=============================================="
echo "📊 GPT数据集可靠性评估"
echo "=============================================="
echo "数据目录: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "使用模型: $GPT_MODEL"
echo "说明: 将自动检查已有结果，跳过已评估的生态"
echo "=============================================="
echo ""

# ---- 设置环境变量 ----
export GPT_MODEL="$GPT_MODEL"
# 如果需要，可以在这里设置API密钥和base_url
# export OPENAI_API_KEY="your-api-key"
# export OPENAI_BASE_URL="your-base-url"

# ---- 运行评估（会自动检查已有结果并跳过） ----
# 不指定 --ecosystem 参数，会评估所有生态
# 代码会自动检查每个生态的结果文件，如果存在则跳过评估
python3 IntentRecBench/src/data_construction/gpt_evaluation.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model "$GPT_MODEL"

echo ""
echo "=============================================="
echo "🎉 所有生态评估完成！"
echo "=============================================="
echo "评估结果保存在: $OUTPUT_DIR"
echo "  - 详细结果: ${OUTPUT_DIR}/{ecosystem}_evaluation_results.json"
echo "  - 汇总统计: ${OUTPUT_DIR}/evaluation_summary.json"
echo "=============================================="

