# TreeRec-TSE

Implementation for “A Needle in a Haystack: Intent-driven Reusable Artifacts Recommendation with LLMs”

## 1. Introduction

This repository provides the official implementation of the paper
“A Needle in a Haystack: Intent-driven Reusable Artifacts Recommendation with LLMs.”

The paper investigates the challenging problem of intent-driven reusable artifacts recommendation, where a developer provides a natural-language intent (e.g., “I need a lightweight HTTP client with retry and timeout support”), and the system automatically identifies the most relevant reusable artifacts such as:
* JavaScript packages
* HuggingFace pretrained models
* Linux distribution components

To achieve this, the paper proposes TreeRec, a semantic-tree-based recommendation framework that leverages large language models (LLMs) to interpret user intent, enrich artifact representations, and enhance ranking quality. The repository also contains the benchmark (RecBench), data processing scripts, semantic tree builder, and end-to-end experimental pipeline.

## 2. Key Features
* Cross-ecosystem support
Works on JavaScript (npm), HuggingFace, and Linux ecosystems.
* Semantic Tree Representation
Converts artifact metadata, descriptions, dependencies, and contextual semantics into hierarchical tree structures.
* LLM-enhanced Recommendation
Uses LLMs to enrich semantic trees, infer missing information, and align user intent with artifact features.
* Unified Benchmark (RecBench)
Includes datasets, evaluation scripts, and metrics for comparing different recommendation strategies.
* Reproducible Experiments
Provides ready-to-run scripts for training, evaluation, and result replication.

3. Repository Structure

```
.
├── data/                      # Raw data & processed benchmark datasets
│   ├── js/                    # JavaScript ecosystem
│   ├── hf/                    # HuggingFace ecosystem
│   └── linux/                 # Linux ecosystem
├── tree_builder/             # Semantic tree construction module
│   ├── parser.py
│   ├── builder.py
│   └── utils.py
├── recommender/              # Recommendation models
│   ├── base_recommender.py
│   ├── tree_enhanced_recommender.py
│   └── llm_prompt_recommender.py
├── experiments/              # Experiment scripts & results
│   ├── train.sh
│   ├── eval.sh
│   └── results/
├── configs/                  # YAML configs for each ecosystem
│   ├── js_config.yaml
│   ├── hf_config.yaml
│   └── linux_config.yaml
├── requirements.txt
└── README.md
```

## 4. Installation

This implementation requires Python 3.8+.
```
git clone https://github.com/jdm4pku/TreeRec-TSE.git
cd TreeRec-TSE
pip install -r requirements.txt
```
Main dependencies include:
* numpy, pandas
* networkx
* torch / transformers (if neural components are enabled)
* pyyaml

## 5. Quick Start
```
bash run_treerec.sh
```
