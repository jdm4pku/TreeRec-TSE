import json
import os
import argparse
import time
from typing import List, Dict, Tuple
from statistics import mean, pstdev
from math import log2

from rank_bm25 import BM25Okapi


# ========== 工具函数 ==========

def load_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, (list, dict)):
        raise ValueError(f"Invalid JSON format in {file_path}")
    return data


def save_json(data, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ========== 评估指标 ==========

# def precision_at_k(ranked_names: List[str], gold: str, k: int) -> float:
#     """计算 P@K（二元相关性）"""
#     top_k = ranked_names[:k]
#     return 1.0 / k if gold in top_k else 0.0

def precision_at_k(ranked_names: List[str], gold: str, k: int) -> float:
    """P@K as Hit@K: 1 if gold appears in top-K, else 0."""
    top_k = ranked_names[:k]
    return 1.0 if gold in top_k else 0.0


def dcg_at_k(ranked_names: List[str], gold: str, k: int) -> float:
    """计算 DCG@K（二元相关性）"""
    for idx, name in enumerate(ranked_names[:k], start=1):
        if name == gold:
            return 1.0 / log2(1.0 + idx)
    return 0.0


# ========== 核心模块 ==========

def build_candidate_docs(candidates: List[Dict], use_fields: List[str]) -> Tuple[List[str], List[str]]:
    """构建候选文档"""
    names, docs = [], []
    for item in candidates:
        names.append(item.get("name", "").strip())
        parts = [item[f] for f in use_fields if f in item and isinstance(item[f], str)]
        docs.append(" \n ".join(parts))
    return names, docs


def build_dataset(dataset: List[Dict]) -> Tuple[List[str], List[str]]:
    """提取意图与对应的正确标签"""
    intents = [row["intent"].strip() for row in dataset]
    gold_labels = [row["artifact"].strip() for row in dataset]
    return intents, gold_labels


def compute_similarity_bm25(intents: List[str], candidate_docs: List[str]) -> Tuple[List[List[int]], List[float]]:
    """
    BM25 相似度计算：
      1. 对文档和查询进行分词
      2. 使用 BM25Okapi 计算每个查询与所有文档的分数
      3. 返回按分数降序排列的索引列表
    """
    # Step 1. 分词
    tokenized_docs = [doc.lower().split() for doc in candidate_docs]
    bm25 = BM25Okapi(tokenized_docs)

    # Step 2. 对每个查询计算分数
    top_indices = []
    query_times: List[float] = []
    for q in intents:
        q_start = time.perf_counter()
        tokenized_query = q.lower().split()
        scores = bm25.get_scores(tokenized_query)
        ranked_idx = scores.argsort()[::-1].tolist()
        top_indices.append(ranked_idx)
        query_times.append(time.perf_counter() - q_start)
    return top_indices, query_times


def evaluate_recommendations(top_indices: List[List[int]], candidate_names: List[str],
                             intents: List[str], gold_labels: List[str],
                             p_ks: List[int], dcg_ks: List[int]):
    """计算 P@K 和 DCG@K"""
    metrics = {f"P@{k}": 0.0 for k in p_ks}
    dcg_metrics = {f"DCG@{k}": 0.0 for k in dcg_ks}
    recommendations = []

    for qi, ranked_idx in enumerate(top_indices):
        ranked_names = [candidate_names[i] for i in ranked_idx]
        rec_entry = {
            "intent": intents[qi],
            "gold": gold_labels[qi],
            "ranking": ranked_names[: max(max(p_ks), max(dcg_ks))],
        }
        recommendations.append(rec_entry)

        for k in p_ks:
            metrics[f"P@{k}"] += precision_at_k(ranked_names, gold_labels[qi], k)
        for k in dcg_ks:
            dcg_metrics[f"DCG@{k}"] += dcg_at_k(ranked_names, gold_labels[qi], k)

    num_q = len(intents)
    for k in p_ks:
        metrics[f"P@{k}"] /= num_q
    for k in dcg_ks:
        dcg_metrics[f"DCG@{k}"] /= num_q

    return metrics, dcg_metrics, recommendations


# ========== 主入口 ==========

def run_bm25_recommendation(data_dir: str, ecosystem: str, output_dir: str,
                            use_fields: List[str], p_ks: List[int], dcg_ks: List[int]):

    start_time = time.time()
    print(f"⏳ Starting BM25 recommendation for [{ecosystem}]...")

    data_path = os.path.join(data_dir, ecosystem)
    dataset = load_json(os.path.join(data_path, "dataset.json"))
    candidates = load_json(os.path.join(data_path, "candidate_artifacts.json"))

    candidate_names, candidate_docs = build_candidate_docs(candidates, use_fields)
    intents, gold_labels = build_dataset(dataset)
    top_indices_per_query, query_times = compute_similarity_bm25(intents, candidate_docs)

    metrics, dcg_metrics, recommendations = evaluate_recommendations(
        top_indices_per_query, candidate_names, intents, gold_labels, p_ks, dcg_ks
    )

    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    query_time_mean = round(mean(query_times), 4) if query_times else 0.0
    query_time_std = round(pstdev(query_times), 4) if len(query_times) > 1 else 0.0
    query_time_min = round(min(query_times), 4) if query_times else 0.0
    query_time_max = round(max(query_times), 4) if query_times else 0.0

    # 保存结果
    output_dir = os.path.join(output_dir, "BM25")
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, f"{ecosystem}-metrics.json")
    rec_path = os.path.join(output_dir, f"{ecosystem}-recommendations.json")
    runtime_summary_path = os.path.join(output_dir, f"{ecosystem}-runtime-summary.txt")

    save_json({
        "precision": metrics,
        "dcg": dcg_metrics,
        "runtime_seconds": elapsed_time,
        "query_time_stats": {
            "mean": query_time_mean,
            "std": query_time_std,
            "min": query_time_min,
            "max": query_time_max
        }
    }, metrics_path)
    save_json(recommendations, rec_path)

    with open(runtime_summary_path, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {ecosystem:<8} | "
                f"Total: {elapsed_time:>8.2f}s | "
                f"Mean: {query_time_mean:>8.4f}s | "
                f"Std: {query_time_std:>8.4f}s | "
                f"Min: {query_time_min:>8.4f}s | "
                f"Max: {query_time_max:>8.4f}s\n")

    print("✅ Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    for k, v in dcg_metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"⏱️ Runtime: {elapsed_time:.2f} seconds")
    print("Query time stats (seconds):")
    print(f"  Mean: {query_time_mean:.4f}")
    print(f"  Std : {query_time_std:.4f}")
    print(f"  Min : {query_time_min:.4f}")
    print(f"  Max : {query_time_max:.4f}")
    print(f"📝 Saved to: {metrics_path}\n")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="IntentRecBench/data")
    parser.add_argument("--ecosystem", type=str, default="hf", choices=["hf", "js", "linux"])
    parser.add_argument("--output_dir", type=str, default="output/baselines")
    parser.add_argument("--use_fields", type=str, nargs="+", default=["name", "type", "description"])
    parser.add_argument("--p_k", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="List of k values for precision calculation")
    parser.add_argument("--dcg_k", type=int, nargs="+", default=[2, 3, 4, 5],
                        help="List of k values for DCG calculation")
    return parser


def main():
    args = get_parser().parse_args()
    run_bm25_recommendation(
        data_dir=args.data_dir,
        ecosystem=args.ecosystem,
        output_dir=args.output_dir,
        use_fields=args.use_fields,
        p_ks=args.p_k,
        dcg_ks=args.dcg_k
    )


if __name__ == "__main__":
    main()