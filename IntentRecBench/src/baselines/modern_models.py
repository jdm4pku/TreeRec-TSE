import os
import json
import time
import argparse
import numpy as np
from math import log2
from typing import List, Dict, Tuple

from sklearn.metrics.pairwise import cosine_similarity

# === 可选依赖 ===
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ========== 工具函数 ==========
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# def precision_at_k(ranked_names, gold, k):
#     return 1.0 / k if gold in ranked_names[:k] else 0.0

def precision_at_k(ranked_names: List[str], gold: str, k: int) -> float:
    """P@K as Hit@K: 1 if gold appears in top-K, else 0."""
    top_k = ranked_names[:k]
    return 1.0 if gold in top_k else 0.0


def dcg_at_k(ranked_names, gold, k):
    for idx, name in enumerate(ranked_names[:k], start=1):
        if name == gold:
            return 1.0 / log2(1.0 + idx)
    return 0.0


# ========== 嵌入模型加载 ==========
# ========== 嵌入模型加载 ==========
def load_embedding_model(model_name: str, provider: str):
    """
    根据 provider 加载模型
    provider 选项：
    - sentence (默认): 从 ModelScope 或 Hugging Face 加载 SentenceTransformer 模型
    - openai: OpenAI embedding API
    """
    if provider == "openai":
        if OpenAI is None:
            raise ImportError("请先安装 openai：pip install openai")
        client = OpenAI()
        return client

    # === SentenceTransformer 模型加载 ===
    if SentenceTransformer is None:
        raise ImportError("请先安装 sentence-transformers：pip install sentence-transformers")

    try:
        # ✅ 优先尝试从 ModelScope 加载
        from modelscope import snapshot_download
        print(f"📦 Downloading model [{model_name}] from ModelScope ...")
        model_dir = snapshot_download(model_name)
        model = SentenceTransformer(model_dir)
        print("✅ Model loaded from ModelScope.")
    except Exception as e:
        print(f"⚠️ ModelScope 加载失败 ({e})，改用 Hugging Face 方式。")
        model = SentenceTransformer(model_name)

    return model


def encode_texts(texts: List[str], model, provider: str):
    """将文本转为向量"""
    if provider == "openai":
        client = model
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=texts
        )
        return np.array([item.embedding for item in response.data])
    else:
        embeddings = model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings)


# ========== 核心流程 ==========
def evaluate(intents, gold_labels, candidate_docs, candidate_names, model, provider, p_ks, dcg_ks):
    intent_vecs = encode_texts(intents, model, provider)
    doc_vecs = encode_texts(candidate_docs, model, provider)

    sims = cosine_similarity(intent_vecs, doc_vecs)
    ranked_indices = np.argsort(-sims, axis=1)

    metrics = {f"P@{k}": 0.0 for k in p_ks}
    dcg_metrics = {f"DCG@{k}": 0.0 for k in dcg_ks}
    recs = []

    for qi, idxs in enumerate(ranked_indices):
        ranked_names = [candidate_names[i] for i in idxs]
        recs.append({
            "intent": intents[qi],
            "gold": gold_labels[qi],
            "ranking": ranked_names[:max(max(p_ks), max(dcg_ks))]
        })
        for k in p_ks:
            metrics[f"P@{k}"] += precision_at_k(ranked_names, gold_labels[qi], k)
        for k in dcg_ks:
            dcg_metrics[f"DCG@{k}"] += dcg_at_k(ranked_names, gold_labels[qi], k)

    num_q = len(intents)
    for k in p_ks:
        metrics[f"P@{k}"] /= num_q
    for k in dcg_ks:
        dcg_metrics[f"DCG@{k}"] /= num_q

    return metrics, dcg_metrics, recs


# ========== 主入口 ==========
def run(data_dir, ecosystem, output_dir, model_name, provider, p_ks, dcg_ks):
    print(f"⏳ Running modern embedding [{model_name}] on ecosystem [{ecosystem}]...")

    dataset = load_json(os.path.join(data_dir, ecosystem, "dataset.json"))
    candidates = load_json(os.path.join(data_dir, ecosystem, "candidate_artifacts.json"))

    intents = [d["intent"].strip() for d in dataset]
    gold_labels = [d["artifact"].strip() for d in dataset]

    candidate_names, candidate_docs = [], []
    for item in candidates:
        candidate_names.append(item.get("name", ""))
        text = " \n ".join([str(v) for v in item.values() if isinstance(v, str)])
        candidate_docs.append(text)

    # ✅ 在计时前加载模型，防止模型加载时间被包含进去
    print("🚀 Loading embedding model...")
    model = load_embedding_model(model_name, provider)
    print("✅ Model loaded successfully.\n")

    # 开始计时（仅统计推理部分）
    start = time.time()

    metrics, dcg_metrics, recs = evaluate(
        intents, gold_labels, candidate_docs, candidate_names,
        model, provider, p_ks, dcg_ks
    )

    elapsed = round(time.time() - start, 2) 
    avg_time_per_query = round(elapsed / len(intents), 4)
    print(f"✅ Evaluation done in {elapsed:.2f}s\n")

    out_dir = os.path.join(output_dir, model_name.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)
    runtime_summary_path = os.path.join(out_dir, f"{ecosystem}-runtime-summary.txt")

    save_json({
        "precision": metrics,
        "dcg": dcg_metrics,
        "runtime_seconds": elapsed,
        "avg_time_per_query": avg_time_per_query
    }, os.path.join(out_dir, f"{ecosystem}-metrics.json"))
    save_json(recs, os.path.join(out_dir, f"{ecosystem}-recommendations.json"))

    with open(runtime_summary_path, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {ecosystem:<8} | "
                f"Total: {elapsed:>8.2f}s | Avg/query: {avg_time_per_query:>8.4f}s\n")

    print("📊 Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    for k, v in dcg_metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"⏱️ Runtime: {elapsed:.2f} seconds")
    print(f"Avg time per query: {avg_time_per_query:.4f} seconds")
    print(f"📝 Saved to: {os.path.join(out_dir, f'{ecosystem}-metrics.json')}\n")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="IntentRecBench/data")
    parser.add_argument("--ecosystem", type=str, default="hf", choices=["hf", "js", "linux"])
    parser.add_argument("--output_dir", type=str, default="output/baselines")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2",choices=["sentence-transformers/all-MiniLM-L6-v2","sentence-transformers/all-mpnet-base-v2"],
                        help="Embedding model name")
    parser.add_argument("--provider", type=str, default="sentence", choices=["sentence", "openai"],
                        help="Embedding provider: 'sentence' for local SentenceTransformer or 'openai'")
    parser.add_argument("--p_k", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--dcg_k", type=int, nargs="+", default=[2, 3, 4, 5])
    return parser


def main():
    args = get_parser().parse_args()
    run(args.data_dir, args.ecosystem, args.output_dir, args.model_name, args.provider, args.p_k, args.dcg_k)


if __name__ == "__main__":
    main()