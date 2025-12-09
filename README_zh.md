# TreeRec

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

**语言**: [English](README.md) | [中文](README_zh.md)

# TreeRec

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

**TreeRec** 是一个基于语义树的意图驱动可重用工件推荐系统，利用大语言模型（LLMs）来理解用户意图、丰富工件表示并提升推荐质量。

## 📖 简介

本项目实现了论文 "A Needle in a Haystack: Intent-driven Reusable Artifacts Recommendation with LLMs" 的官方代码。

TreeRec 解决了意图驱动的可重用工件推荐这一挑战性问题。当开发者提供自然语言意图描述（例如："我需要一个支持重试和超时的轻量级 HTTP 客户端"）时，系统能够自动识别最相关的可重用工件，包括：

- **JavaScript 包** (npm)
- **HuggingFace 预训练模型**
- **Linux 发行版组件**

### 核心特性

- 🌳 **语义树表示**：将工件元数据、描述、依赖关系和上下文语义转换为层次化树结构
- 🤖 **LLM 增强推荐**：使用大语言模型丰富语义树、推断缺失信息，并将用户意图与工件特征对齐
- 🔄 **跨生态系统支持**：支持 JavaScript (npm)、HuggingFace 和 Linux 生态系统
- 📊 **统一基准测试 (IntentRecBench)**：包含数据集、评估脚本和指标，用于比较不同的推荐策略
- 🔬 **可复现实验**：提供即用型脚本用于训练、评估和结果复现

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA（如果使用 GPU 加速）

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/TreeRec.git
cd TreeRec

# 安装依赖
pip install -r requirements.txt
```

### 配置 API 密钥

TreeRec 需要配置 LLM API 密钥。请通过环境变量设置：

```bash
# OpenAI API（如果使用 OpenAI 模型）
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"

# 或者使用其他 API 服务（如 SiliconFlow、OpenRouter 等）
export OPENAI_BASE_URL="https://api.siliconflow.cn/v1"
export OPENAI_API_KEY="your-api-key"
```

**注意**：某些模型可能需要额外的 API 密钥环境变量，请参考具体模型的文档。

### 运行示例

```bash
# 运行 TreeRec 推荐（使用默认配置）
bash run_treerec.sh

# 或使用 Python 脚本
python run_treerec.py \
    --data_dir IntentRecBench/data \
    --ecosystem js \
    --llm_name Qwen3-8B \
    --rerank_model Qwen/Qwen3-8B \
    --use_rerank \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
    --summarization_model Qwen/Qwen3-8B
```

## 📁 项目结构

```
TreeRec/
├── TreeRec/                      # 核心模块
│   ├── tree_builder.py          # 语义树构建器
│   ├── tree_retriever.py        # 树检索器
│   ├── cluster_tree_builder.py  # 聚类树构建器
│   ├── RetrievalAugmentation.py # 检索增强主类
│   ├── EmbeddingModels.py       # 嵌入模型封装
│   ├── RerankModels.py          # 重排序模型封装
│   ├── SummarizationModels.py   # 摘要模型封装
│   ├── Retrievers.py            # 检索器基类
│   └── utils.py                 # 工具函数
├── IntentRecBench/              # 基准测试数据集和评估
│   ├── data/                    # 数据集
│   │   ├── js/                  # JavaScript 生态系统数据
│   │   ├── hf/                  # HuggingFace 生态系统数据
│   │   └── linux/               # Linux 生态系统数据
│   ├── src/                     # 源代码
│   │   ├── baselines/           # 基线方法实现
│   │   └── data_construction/   # 数据构建脚本
│   └── script/                  # 运行脚本
├── prompt/                      # 提示词模板
│   ├── summarization_system.txt # 摘要系统提示词
│   ├── summarization_user.txt   # 摘要用户提示词
│   ├── rerank_system.txt        # 重排序系统提示词
│   └── rerank_user.txt          # 重排序用户提示词
├── output/                      # 输出结果目录
├── exp_figure/                  # 实验图表生成脚本
├── run_treerec.py              # 主运行脚本
├── run_treerec.sh              # Shell 运行脚本
├── case_study.py               # 案例研究脚本
├── test.py                     # 测试/工具脚本
├── requirements.txt            # Python 依赖
└── README.md                   # 本文件
```

## 🔧 使用方法

### 基本用法

```python
from TreeRec.RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig

# 创建配置
config = RetrievalAugmentationConfig(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    summarization_model="Qwen/Qwen3-8B",
    rerank_model="Qwen/Qwen3-8B",
    use_rerank=True,
    tree_builder_type="cluster"
)

# 初始化检索增强系统
ra = RetrievalAugmentation(config=config)

# 添加工件
artifacts = [
    {"name": "axios", "description": "Promise based HTTP client"},
    {"name": "lodash", "description": "Modern JavaScript utility library"},
    # ... 更多工件
]
ra.add_artifacts(artifacts)

# 保存树结构（可选）
ra.save("output/tree.pkl")

# 进行推荐
intent = "I need a lightweight HTTP client with retry and timeout support"
recommendations = ra.artifact_recommendation(intent, top_k=5)
print(recommendations)
```

### 命令行参数

主要参数说明：

- `--data_dir`: 数据目录路径（默认：`IntentRecBench/data`）
- `--ecosystem`: 生态系统，可选 `js`、`hf`、`linux`
- `--llm_name`: LLM 模型名称（用于输出目录命名）
- `--rerank_model`: 重排序模型名称
- `--use_rerank`: 是否使用重排序模型
- `--embedding_model`: 嵌入模型名称
- `--summarization_model`: 摘要模型名称
- `--tree_builder_type`: 树构建器类型（默认：`cluster`）
- `--top_k`: 返回前 K 个推荐结果（默认：5）
- `--p_k`: Precision@K 评估的 K 值列表（默认：`1 2 3 4`）
- `--dcg_k`: DCG@K 评估的 K 值列表（默认：`2 3 4 5`）

完整参数列表请运行：

```bash
python run_treerec.py --help
```

## 📊 评估指标

TreeRec 使用以下指标评估推荐性能：

- **Precision@K (P@K)**：前 K 个推荐中正确工件的命中率
- **DCG@K (Discounted Cumulative Gain@K)**：考虑排序位置的累积增益

结果保存在 `output/TreeRec/<llm_name>/<ecosystem>-metrics.json`。

## 🧪 实验复现

### 运行基线方法

```bash
# 运行 BM25 基线
bash IntentRecBench/script/run_bm25.sh

# 运行 TF-IDF 基线
bash IntentRecBench/script/run_tf_idf.sh

# 运行其他基线方法...
```

### 运行 TreeRec

```bash
# 使用不同配置运行 TreeRec
bash run_treerec.sh      # 默认配置
bash run_treerec2.sh     # 配置 2
# ... 更多配置
```

## 🔍 案例研究

运行案例研究脚本查看详细推荐过程：

```bash
python case_study.py \
    --data_dir IntentRecBench/data \
    --ecosystem js \
    --intent "I want to easily add and self-host the Montserrat font in my web project using npm."
```

## 📝 依赖项

主要依赖包括：

- `torch` - PyTorch 深度学习框架
- `transformers` - HuggingFace Transformers 库
- `sentence-transformers` - 句子嵌入模型
- `openai` - OpenAI API 客户端
- `faiss-cpu` - 相似度搜索库
- `scikit-learn` - 机器学习工具
- `numpy` - 数值计算
- `tiktoken` - Token 计数
- `tenacity` - 重试机制

完整列表请查看 `requirements.txt`。

## ⚠️ 注意事项

1. **API 密钥安全**：请勿在代码中硬编码 API 密钥，始终使用环境变量
2. **资源消耗**：构建大型语义树可能需要较长时间和大量内存
3. **模型选择**：不同模型在性能和成本之间有不同的权衡，请根据需求选择
4. **数据格式**：确保输入数据符合预期格式（包含 `name` 和 `description` 字段）

## 🤝 贡献

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件至项目维护者

## 🙏 致谢

感谢所有为本项目做出贡献的研究者和开发者。

---

**注意**：本项目为研究用途，使用前请确保遵守相关 API 服务的使用条款和限制。
