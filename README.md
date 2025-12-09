# TreeRec

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

**TreeRec** is a semantic-tree-based intent-driven reusable artifacts recommendation system that leverages large language models (LLMs) to interpret user intent, enrich artifact representations, and enhance ranking quality.

**Language**: [English](README.md) | [中文](README_zh.md)

## 📖 Introduction

This repository provides the official implementation of the paper **"A Needle in a Haystack: Intent-driven Reusable Artifacts Recommendation with LLMs"**.

TreeRec addresses the challenging problem of intent-driven reusable artifacts recommendation. When a developer provides a natural-language intent description (e.g., "I need a lightweight HTTP client with retry and timeout support"), the system automatically identifies the most relevant reusable artifacts, including:

- **JavaScript packages** (npm)
- **HuggingFace pretrained models**
- **Linux distribution components**

### Key Features

- 🌳 **Semantic Tree Representation**: Converts artifact metadata, descriptions, dependencies, and contextual semantics into hierarchical tree structures
- 🤖 **LLM-enhanced Recommendation**: Uses large language models to enrich semantic trees, infer missing information, and align user intent with artifact features
- 🔄 **Cross-ecosystem Support**: Works on JavaScript (npm), HuggingFace, and Linux ecosystems
- 📊 **Unified Benchmark (IntentRecBench)**: Includes datasets, evaluation scripts, and metrics for comparing different recommendation strategies
- 🔬 **Reproducible Experiments**: Provides ready-to-run scripts for training, evaluation, and result replication

## 🚀 Quick Start

### Requirements

- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/TreeRec.git
cd TreeRec

# Install dependencies
pip install -r requirements.txt
```

### API Key Configuration

TreeRec requires LLM API keys to be configured. Set them via environment variables:

```bash
# OpenAI API (if using OpenAI models)
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"

# Or use other API services (e.g., SiliconFlow, OpenRouter)
export OPENAI_BASE_URL="https://api.siliconflow.cn/v1"
export OPENAI_API_KEY="your-api-key"
```

**Note**: Some models may require additional API key environment variables. Please refer to the specific model documentation.

### Running Examples

```bash
# Run TreeRec recommendation (using default configuration)
bash run_treerec.sh

# Or use Python script directly
python run_treerec.py \
    --data_dir IntentRecBench/data \
    --ecosystem js \
    --llm_name Qwen3-8B \
    --rerank_model Qwen/Qwen3-8B \
    --use_rerank \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
    --summarization_model Qwen/Qwen3-8B
```

## 📁 Project Structure

```
TreeRec/
├── TreeRec/                      # Core modules
│   ├── tree_builder.py          # Semantic tree builder
│   ├── tree_retriever.py        # Tree retriever
│   ├── cluster_tree_builder.py  # Cluster-based tree builder
│   ├── RetrievalAugmentation.py # Main retrieval augmentation class
│   ├── EmbeddingModels.py       # Embedding model wrappers
│   ├── RerankModels.py          # Reranking model wrappers
│   ├── SummarizationModels.py   # Summarization model wrappers
│   ├── Retrievers.py            # Retriever base classes
│   └── utils.py                 # Utility functions
├── IntentRecBench/              # Benchmark datasets and evaluation
│   ├── data/                    # Datasets
│   │   ├── js/                  # JavaScript ecosystem data
│   │   ├── hf/                  # HuggingFace ecosystem data
│   │   └── linux/               # Linux ecosystem data
│   ├── src/                     # Source code
│   │   ├── baselines/           # Baseline method implementations
│   │   └── data_construction/   # Data construction scripts
│   └── script/                  # Execution scripts
├── prompt/                      # Prompt templates
│   ├── summarization_system.txt # Summarization system prompt
│   ├── summarization_user.txt   # Summarization user prompt
│   ├── rerank_system.txt        # Reranking system prompt
│   └── rerank_user.txt          # Reranking user prompt
├── output/                      # Output results directory
├── exp_figure/                  # Experiment figure generation scripts
├── run_treerec.py              # Main execution script
├── run_treerec.sh              # Shell execution script
├── case_study.py               # Case study script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Usage

### Basic Usage

```python
from TreeRec.RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig

# Create configuration
config = RetrievalAugmentationConfig(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    summarization_model="Qwen/Qwen3-8B",
    rerank_model="Qwen/Qwen3-8B",
    use_rerank=True,
    tree_builder_type="cluster"
)

# Initialize retrieval augmentation system
ra = RetrievalAugmentation(config=config)

# Add artifacts
artifacts = [
    {"name": "axios", "description": "Promise based HTTP client"},
    {"name": "lodash", "description": "Modern JavaScript utility library"},
    # ... more artifacts
]
ra.add_artifacts(artifacts)

# Save tree structure (optional)
ra.save("output/tree.pkl")

# Perform recommendation
intent = "I need a lightweight HTTP client with retry and timeout support"
recommendations = ra.artifact_recommendation(intent, top_k=5)
print(recommendations)
```

### Command Line Arguments

Main parameter descriptions:

- `--data_dir`: Path to data directory (default: `IntentRecBench/data`)
- `--ecosystem`: Ecosystem name, options: `js`, `hf`, `linux`
- `--llm_name`: LLM model name (used for output directory naming)
- `--rerank_model`: Reranking model name
- `--use_rerank`: Whether to use reranking model
- `--embedding_model`: Embedding model name
- `--summarization_model`: Summarization model name
- `--tree_builder_type`: Tree builder type (default: `cluster`)
- `--top_k`: Return top K recommendation results (default: 5)
- `--p_k`: K values for Precision@K evaluation (default: `1 2 3 4`)
- `--dcg_k`: K values for DCG@K evaluation (default: `2 3 4 5`)

For complete parameter list, run:

```bash
python run_treerec.py --help
```

## 📊 Evaluation Metrics

TreeRec uses the following metrics to evaluate recommendation performance:

- **Precision@K (P@K)**: Hit rate of correct artifacts in top K recommendations
- **DCG@K (Discounted Cumulative Gain@K)**: Cumulative gain considering ranking positions

Results are saved in `output/TreeRec/<llm_name>/<ecosystem>-metrics.json`.

## 🧪 Reproducing Experiments

### Running Baseline Methods

```bash
# Run BM25 baseline
bash IntentRecBench/script/run_bm25.sh

# Run TF-IDF baseline
bash IntentRecBench/script/run_tf_idf.sh

# Run other baseline methods...
```

### Running TreeRec

```bash
# Run TreeRec with different configurations
bash run_treerec.sh      # Default configuration
bash run_treerec2.sh     # Configuration 2
# ... more configurations
```

## 🔍 Case Study

Run the case study script to view detailed recommendation process:

```bash
python case_study.py \
    --data_dir IntentRecBench/data \
    --ecosystem js \
    --intent "I want to easily add and self-host the Montserrat font in my web project using npm."
```

## 📝 Dependencies

Main dependencies include:

- `torch` - PyTorch deep learning framework
- `transformers` - HuggingFace Transformers library
- `sentence-transformers` - Sentence embedding models
- `openai` - OpenAI API client
- `faiss-cpu` - Similarity search library
- `scikit-learn` - Machine learning utilities
- `numpy` - Numerical computing
- `tiktoken` - Token counting
- `tenacity` - Retry mechanism

See `requirements.txt` for the complete list.

## ⚠️ Important Notes

1. **API Key Security**: Never hardcode API keys in code. Always use environment variables
2. **Resource Consumption**: Building large semantic trees may require significant time and memory
3. **Model Selection**: Different models have different trade-offs between performance and cost. Choose according to your needs
4. **Data Format**: Ensure input data conforms to expected format (contains `name` and `description` fields)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or suggestions, please contact us via:

- Opening an Issue
- Sending an email to the project maintainers

## 🙏 Acknowledgments

Thanks to all researchers and developers who have contributed to this project.

---

**Note**: This project is for research purposes. Please ensure compliance with relevant API service terms and limitations before use.

