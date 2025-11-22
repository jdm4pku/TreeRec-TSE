import os
import json
import time
import argparse
from typing import List, Dict
from math import log2
from statistics import mean, pstdev

os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = ""

from TreeRec.RetrievalAugmentation import RetrievalAugmentation
from TreeRec.RetrievalAugmentation import RetrievalAugmentationConfig


# ============================================================
# 工具函数
# ============================================================

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


# ============================================================
# 评估指标
# ============================================================

# def precision_at_k(ranked_names: List[str], gold: str, k: int) -> float:
#     top_k = ranked_names[:k]
#     return 1.0 / k if gold in top_k else 0.0

def precision_at_k(ranked_names: List[str], gold: str, k: int) -> float:
    """P@K as Hit@K: 1 if gold appears in top-K, else 0."""
    top_k = ranked_names[:k]
    return 1.0 if gold in top_k else 0.0

def dcg_at_k(ranked_names: List[str], gold: str, k: int) -> float:
    for idx, name in enumerate(ranked_names[:k], start=1):
        if name == gold:
            return 1.0 / log2(1.0 + idx)
    return 0.0


def evaluate_recommendations(top_names: List[List[str]], candidate_names: List[str],
                             intents: List[str], gold_labels: List[str],
                             p_ks: List[int], dcg_ks: List[int]):
    metrics = {f"P@{k}": 0.0 for k in p_ks}
    dcg_metrics = {f"DCG@{k}": 0.0 for k in dcg_ks}
    recommendations = []

    for qi, ranked_names in enumerate(top_names):
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


# ============================================================
# 主流程：TreeRec 检索与评估
# ============================================================

def run_treerec(args):
    data_dir = args.data_dir
    ecosystem = args.ecosystem
    output_dir = args.output_dir
    top_k = args.top_k
    p_ks = args.p_k
    dcg_ks = args.dcg_k
    tree_path = args.tree_path

    start_time = time.time()
    print(f"⏳ Starting TreeRec recommendation for [{ecosystem}]...")

    # ---------- 加载数据 ----------
    data_path = os.path.join(data_dir, ecosystem)
    dataset = load_json(os.path.join(data_path, "dataset.json"))
    candidates = load_json(os.path.join(data_path, "candidate_artifacts.json"))

    candidate_names = [c["name"] for c in candidates if "name" in c]
    intents = [row["intent"].strip() for row in dataset]
    gold_labels = [row["artifact"].strip() for row in dataset]

    # ---------- 构建/加载 TreeRec ----------
    tree_dir = os.path.join(output_dir, "TreeRec")
    os.makedirs(tree_dir, exist_ok=True)
    tree_dir = os.path.join(tree_dir, args.llm_name)
    os.makedirs(tree_dir, exist_ok=True)
    default_tree_path = os.path.join(tree_dir, f"{ecosystem}-tree.pkl")
    tree_path = tree_path or default_tree_path

    # 创建配置（构建和加载都需要）
    config = RetrievalAugmentationConfig(
        tree_builder_config=None,
        tree_retriever_config=None,
        rerank_model=args.rerank_model,
        embedding_model=args.embedding_model,
        summarization_model=args.summarization_model,
        tree_builder_type=args.tree_builder_type,
        use_rerank=args.use_rerank,
        # TreeRetrieverConfig arguments
        tr_tokenizer=args.tr_tokenizer,
        tr_threshold=args.tr_threshold,
        tr_top_k=args.tr_top_k,
        tr_selection_mode=args.tr_selection_mode,
        tr_context_embedding_model=args.tr_context_embedding_model,
        tr_embedding_model=args.tr_embedding_model,
        tr_num_layers=args.tr_num_layers,
        tr_start_layer=args.tr_start_layer,
        # TreeBuilderConfig arguments
        tb_tokenizer=args.tb_tokenizer,
        tb_max_tokens=args.tb_max_tokens,
        tb_num_layers=args.tb_num_layers,
        tb_threshold=args.tb_threshold,
        tb_top_k=args.tb_top_k,
        tb_selection_mode=args.tb_selection_mode,
        tb_summarization_length=args.tb_summarization_length,
        tb_summarization_model=args.tb_summarization_model,
        tb_embedding_model=args.tb_embedding_model,
        tb_cluster_embedding_model=args.tb_cluster_embedding_model,
    )

    if os.path.exists(tree_path):
        load_start = time.time()
        RA = RetrievalAugmentation(config=config, tree=tree_path)
        load_end = time.time()
        build_tree_seconds = round(load_end - load_start, 2)
        print(f"🌲 Tree loaded for [{ecosystem}] from {tree_path} in {build_tree_seconds:.2f}s")
    else:
        build_start = time.time()
        RA = RetrievalAugmentation(config=config)
        RA.add_artifacts(candidates)
        RA.save(tree_path)
        build_end = time.time()
        build_tree_seconds = round(build_end - build_start, 2)
        print(f"🌲 Tree built and saved for [{ecosystem}] to {tree_path} in {build_tree_seconds:.2f}s")

    # ---------- 检索 ----------
    retrieval_start = time.time()
    all_rankings = []
    query_times = []
    for intent in intents:
        query_start = time.time()
        top_k_artifacts = RA.artifact_recommendation(intent, top_k=top_k)
        query_end = time.time()
        query_times.append(query_end - query_start)
        all_rankings.append(top_k_artifacts)
    retrieval_end = time.time()

    retrieval_seconds = round(retrieval_end - retrieval_start, 2)
    avg_retrieval_seconds = round(mean(query_times), 4) if query_times else 0.0
    std_retrieval_seconds = round(pstdev(query_times), 4) if len(query_times) > 1 else 0.0
    min_retrieval_seconds = round(min(query_times), 4) if query_times else 0.0
    max_retrieval_seconds = round(max(query_times), 4) if query_times else 0.0

    # ---------- 评估 ----------
    metrics, dcg_metrics, recommendations = evaluate_recommendations(
        all_rankings, candidate_names, intents, gold_labels, p_ks, dcg_ks
    )

    end_time = time.time()
    total_seconds = round(end_time - start_time, 2)
    avg_total_seconds = round(total_seconds / len(intents), 4)

    # ---------- 保存结果 ----------
    output_dir = os.path.join(output_dir, "TreeRec")
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, args.llm_name)
    os.makedirs(output_dir, exist_ok=True)
    # 在文件名中包含 rerank 信息以便区分消融实验结果
    rerank_suffix = "rerank" if args.use_rerank else "no-rerank"
    metrics_path = os.path.join(output_dir, f"{ecosystem}-{rerank_suffix}-metrics.json")
    rec_path = os.path.join(output_dir, f"{ecosystem}-{rerank_suffix}-recommendations.json")
    runtime_summary_path = os.path.join(output_dir, f"{ecosystem}-{rerank_suffix}-runtime-summary.txt")

    save_json({
        "precision": metrics,
        "dcg": dcg_metrics,
        "time": {
            "build_tree_seconds": build_tree_seconds,
            "retrieval_seconds": retrieval_seconds,
            "avg_retrieval_seconds": avg_retrieval_seconds,
            "std_retrieval_seconds": std_retrieval_seconds,
            "min_retrieval_seconds": min_retrieval_seconds,
            "max_retrieval_seconds": max_retrieval_seconds,
            "total_seconds": total_seconds,
            "avg_total_seconds": avg_total_seconds
        }
    }, metrics_path)

    save_json(recommendations, rec_path)

    with open(runtime_summary_path, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {ecosystem:<8} | "
                f"Tree: {build_tree_seconds:>7.2f}s | Retrieval: {retrieval_seconds:>7.2f}s | "
                f"Total: {total_seconds:>7.2f}s | AvgRetr: {avg_retrieval_seconds:>7.4f}s | "
                f"StdRetr: {std_retrieval_seconds:>7.4f}s | MinRetr: {min_retrieval_seconds:>7.4f}s | "
                f"MaxRetr: {max_retrieval_seconds:>7.4f}s | AvgTotal: {avg_total_seconds:>7.4f}s\n")

    # ---------- 输出日志 ----------
    print("✅ Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    for k, v in dcg_metrics.items():
        print(f"{k}: {v:.4f}")
    print("\n⏱️ Runtime Summary:")
    print(f"  Tree Build Time:        {build_tree_seconds:.2f}s")
    print(f"  Retrieval Time:         {retrieval_seconds:.2f}s")
    print(f"  Avg Retrieval / Query:  {avg_retrieval_seconds:.4f}s")
    print(f"  Std Retrieval / Query:  {std_retrieval_seconds:.4f}s")
    print(f"  Min Retrieval / Query:  {min_retrieval_seconds:.4f}s")
    print(f"  Max Retrieval / Query:  {max_retrieval_seconds:.4f}s")
    print(f"  Total Time:             {total_seconds:.2f}s")
    print(f"  Avg Total / Query:      {avg_total_seconds:.4f}s")
    print(f"📝 Saved to: {metrics_path}\n")


# ============================================================
# 命令行接口
# ============================================================

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="IntentRecBench/data", help="Path to the data directory")
    parser.add_argument("--llm_name", type=str, default="Qwen3_14B", help="Name of the LLM")
    parser.add_argument("--ecosystem", type=str, default="linux", choices=["hf", "js", "linux"],help="Name of the ecosystem")
    parser.add_argument("--output_dir", type=str, default="output",help="Path to the output directory")
    # model setting
    parser.add_argument("--rerank_model", type=str, default="Qwen/Qwen3-14B",help="Rerank model name")
    parser.add_argument("--use_rerank", action="store_true", help="Whether to use rerank model")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",help="Embedding model name")
    parser.add_argument("--summarization_model", type=str, default="Qwen/Qwen3-14B",help="Summarization model name")
    parser.add_argument("--tree_builder_type", type=str, default="cluster",help="Tree builder type")
    # TreeRetrieverConfig arguments
    parser.add_argument("--tr_tokenizer", type=str, default=None,help="Tokenizer name")
    parser.add_argument("--tr_threshold", type=float, default=0.5,help="Threshold for tree retrieval")
    parser.add_argument("--tr_top_k", type=int, default=5,help="Top k for tree retrieval")
    parser.add_argument("--tr_selection_mode", type=str, default="top_k",help="Selection mode for tree retrieval")
    parser.add_argument("--tr_context_embedding_model", type=str, help="Context embedding model name")
    parser.add_argument("--tr_embedding_model", type=str, help="Embedding model name")
    parser.add_argument("--tr_num_layers", type=int, default=None, help="Number of layers for tree retrieval")
    parser.add_argument("--tr_start_layer", type=int, default=None, help="Start layer for tree retrieval")
    # TreeBuilderConfig arguments
    parser.add_argument("--tb_tokenizer", type=str, default=None,help="Tokenizer name")
    parser.add_argument("--tb_max_tokens", type=int, default=100, help="Max tokens for tree builder")
    parser.add_argument("--tb_num_layers", type=int, default=5, help="Number of layers for tree builder")
    parser.add_argument("--tb_threshold", type=float, default=0.5, help="Threshold for tree builder")
    parser.add_argument("--tb_top_k", type=int, default=5, help="Top k for tree builder")
    parser.add_argument("--tb_selection_mode", type=str, default="top_k", help="Selection mode for tree builder")
    parser.add_argument("--tb_summarization_length", type=int, default=100, help="Summarization length for tree builder")
    parser.add_argument("--tb_summarization_model", type=str, help="Summarization model name")
    parser.add_argument("--tb_embedding_model", type=str, help="Embedding model name")
    parser.add_argument("--tb_cluster_embedding_model", type=str, help="Cluster embedding model name")
    # output setting
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--p_k", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--dcg_k", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--tree_path", type=str, default=None,
                        help="Path to load/save the tree pickle. Defaults to output_dir/TreeRec/<ecosystem>-tree.pkl")
    return parser


def main():
    args = get_parser().parse_args()
    run_treerec(args)


if __name__ == "__main__":
    main()