import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

os.environ["OPENAI_API_KEY"] = "sk-BwTI1iSg83soUQ6u2d1096B8A27848E5B3E4141154Dc592b"
os.environ["OPENAI_BASE_URL"] = "http://66.206.9.230:4000/v1"

from TreeRec.RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig
from TreeRec.utils import reverse_mapping


DEFAULT_INTENT = "I want to easily add and self-host the Montserrat font in my web project using npm."


def load_json(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON 文件不存在：{file_path}")
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, (list, dict)):
        raise ValueError(f"Invalid JSON format in {file_path}")
    return data


def save_json(data, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def summarize_text(text: Optional[str], max_len: int = 500) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def build_parent_map(tree) -> Dict[int, List[int]]:
    parent_map: Dict[int, List[int]] = {}
    for parent_idx, node in tree.all_nodes.items():
        for child_idx in node.children:
            parent_map.setdefault(child_idx, []).append(parent_idx)
    for child_idx in parent_map:
        parent_map[child_idx] = sorted(set(parent_map[child_idx]))
    return parent_map


def build_path_to_root(node_idx: int, parent_map: Dict[int, List[int]], node_to_layer: Dict[int, int], tree) -> List[Dict[str, Any]]:
    path: List[Dict[str, Any]] = []
    current = node_idx
    visited = set()

    while True:
        node = tree.all_nodes.get(current)
        if node is None:
            break
        path.append(
            {
                "node_index": current,
                "layer": node_to_layer.get(current),
                "name": node.name,
            }
        )
        parents = parent_map.get(current, [])
        if not parents:
            break
        parent = parents[0]
        if parent in visited:
            break
        visited.add(parent)
        current = parent

    return list(reversed(path))


def find_rerank_entry(intent: str, rerank_path: str, top_k: int) -> Dict[str, Any]:
    rerank_data = load_json(rerank_path)
    if not isinstance(rerank_data, list):
        raise ValueError(f"rerank 文件格式错误：{rerank_path}")
    normalized = intent.strip()
    for entry in rerank_data:
        if entry.get("intent", "").strip() == normalized:
            ranking = entry.get("ranking", [])[:top_k]
            return {
                "gold": entry.get("gold"),
                "ranking": ranking,
            }
    raise ValueError(f"未在 {rerank_path} 中找到意图：{intent}")


def upsert_case_entry(entry: Dict[str, Any], output_path: str):
    if os.path.exists(output_path):
        current = load_json(output_path)
        if not isinstance(current, list):
            raise ValueError(f"{output_path} 不是 list 格式，无法写入新的 case study。")
    else:
        current = []

    filtered = [
        e for e in current
        if not (e.get("ecosystem") == entry["ecosystem"] and e.get("intent") == entry["intent"])
    ]
    filtered.append(entry)
    save_json(filtered, output_path)


def build_ra(tree_path: str, args) -> RetrievalAugmentation:
    config = RetrievalAugmentationConfig(
        rerank_model=args.rerank_model,
        embedding_model=args.embedding_model,
        summarization_model=args.summarization_model,
        tree_builder_type=args.tree_builder_type,
        use_rerank=False,  # rerank 结果直接来自离线文件
        tr_threshold=args.tr_threshold,
        tr_top_k=args.tr_top_k,
        tr_selection_mode=args.tr_selection_mode,
        tr_context_embedding_model=args.tr_context_embedding_model or args.embedding_model,
        tr_embedding_model=args.tr_embedding_model or args.embedding_model,
        tr_num_layers=args.tr_num_layers,
        tr_start_layer=args.tr_start_layer,
        tb_max_tokens=args.tb_max_tokens,
        tb_num_layers=args.tb_num_layers,
        tb_threshold=args.tb_threshold,
        tb_top_k=args.tb_top_k,
        tb_selection_mode=args.tb_selection_mode,
        tb_summarization_length=args.tb_summarization_length,
        tb_summarization_model=args.tb_summarization_model or args.summarization_model,
        tb_embedding_model=args.tb_embedding_model or args.embedding_model,
        tb_cluster_embedding_model=args.tb_cluster_embedding_model or args.embedding_model,
    )
    return RetrievalAugmentation(config=config, tree=tree_path)


def resolve_paths(args):
    base_dir = os.path.abspath(os.path.join(args.base_output_dir, args.llm_name))
    tree_path = os.path.abspath(args.tree_path) if args.tree_path else os.path.join(base_dir, f"{args.ecosystem}-tree.pkl")
    rerank_path = os.path.abspath(args.rerank_file) if args.rerank_file else os.path.join(base_dir, f"{args.ecosystem}-rerank-recommendations.json")
    case_output_path = os.path.abspath(args.case_output) if args.case_output else os.path.join(base_dir, "case_studies.json")
    return tree_path, rerank_path, case_output_path


def run_case_study(args):
    tree_path, rerank_path, case_output_path = resolve_paths(args)

    if not os.path.exists(tree_path):
        raise FileNotFoundError(f"Tree 文件不存在：{tree_path}")
    if not os.path.exists(rerank_path):
        raise FileNotFoundError(f"推荐结果文件不存在：{rerank_path}")

    ra = build_ra(tree_path, args)
    tree = ra.tree
    node_to_layer = reverse_mapping(tree.layer_to_nodes)
    parent_map = build_parent_map(tree)

    print(f"🌲 已加载树：{tree_path}")
    print(f"🔍 开始针对意图进行 TreeRec 检索：{args.intent}")

    retrieval_start = time.time()
    selected_nodes, context, layer_information = ra.retrieve(
        args.intent,
        top_k=args.retrieval_top_k,
        collapse_tree=args.collapse_tree,
        return_layer_information=True,
    )
    retrieval_time = round(time.time() - retrieval_start, 4)

    retrieved_candidates = [node.name for node in selected_nodes if node.name]

    node_details = []
    paths = []
    for node, layer_meta in zip(selected_nodes, layer_information):
        node_details.append(
            {
                "node_index": node.index,
                "layer": layer_meta.get("layer_number"),
                "name": node.name,
                "desc_preview": summarize_text(node.desc, max_len=600),
            }
        )
        paths.append(
            {
                "candidate": node.name,
                "node_index": node.index,
                "path": build_path_to_root(node.index, parent_map, node_to_layer, tree),
            }
        )

    rerank_entry = find_rerank_entry(args.intent, rerank_path, args.rerank_top_k)
    reranked_candidates = rerank_entry["ranking"]
    predicted_candidate = reranked_candidates[0] if reranked_candidates else (retrieved_candidates[0] if retrieved_candidates else "")

    case_entry = {
        "ecosystem": args.ecosystem,
        "intent": args.intent,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tree_path": tree_path,
        "recommendation_source": rerank_path,
        "predicted_candidate": predicted_candidate,
        "gold": rerank_entry.get("gold"),
        "retrieval": {
            "top_k": args.retrieval_top_k,
            "retrieval_time_seconds": retrieval_time,
            "collapse_tree": args.collapse_tree,
            "retrieved_candidates": retrieved_candidates,
            "paths": paths,
            "node_details": node_details,
            "context_preview": summarize_text(context, max_len=1800),
        },
        "rerank": {
            "top_k": args.rerank_top_k,
            "candidates": reranked_candidates,
            "source": rerank_path,
        },
    }

    upsert_case_entry(case_entry, case_output_path)

    print("✅ Case study 已更新：")
    print(f"   - 预测结果: {predicted_candidate}")
    print(f"   - Gold: {case_entry['gold']}")
    print(f"   - 检索节点数: {len(node_details)}")
    print(f"   - 重排列表长度: {len(reranked_candidates)}")
    print(f"📄 写入路径: {case_output_path}")


def get_parser():
    parser = argparse.ArgumentParser(description="TreeRec case study 工具")
    parser.add_argument("--intent", type=str, default=DEFAULT_INTENT, help="需要分析的意图描述")
    parser.add_argument("--ecosystem", type=str, default="js", choices=["js"], help="生态名称，目前仅支持 js")
    parser.add_argument("--llm_name", type=str, default="gpt4o", help="TreeRec 运行时使用的 LLM 名称（用于定位输出目录）")
    parser.add_argument("--base_output_dir", type=str, default="output/TreeRec", help="TreeRec 输出基目录")
    parser.add_argument("--tree_path", type=str, default=None, help="自定义树文件路径，若为空则根据 llm_name/ecosystem 推断")
    parser.add_argument("--rerank_file", type=str, default=None, help="rerank 推荐结果文件路径")
    parser.add_argument("--case_output", type=str, default=None, help="case study JSON 输出路径")
    parser.add_argument("--retrieval_top_k", type=int, default=10, help="检索阶段的 top-k")
    parser.add_argument("--rerank_top_k", type=int, default=10, help="读取 rerank 结果时截取的 top-k")
    parser.add_argument("--collapse_tree", action="store_true", help="是否在检索时对树进行合并检索（collapse_tree 模式）")

    # TreeRetriever / TreeBuilder 相关参数（与 run_treerec 保持一致，必要时可覆盖）
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--summarization_model", type=str, default="gpt-4o-2024-05-13")
    parser.add_argument("--rerank_model", type=str, default="gpt-4o-2024-05-13")
    parser.add_argument("--tree_builder_type", type=str, default="cluster")
    parser.add_argument("--tr_threshold", type=float, default=0.5)
    parser.add_argument("--tr_top_k", type=int, default=40)
    parser.add_argument("--tr_selection_mode", type=str, default="top_k")
    parser.add_argument("--tr_context_embedding_model", type=str, default=None)
    parser.add_argument("--tr_embedding_model", type=str, default=None)
    parser.add_argument("--tr_num_layers", type=int, default=None)
    parser.add_argument("--tr_start_layer", type=int, default=None)
    parser.add_argument("--tb_max_tokens", type=int, default=100)
    parser.add_argument("--tb_num_layers", type=int, default=5)
    parser.add_argument("--tb_threshold", type=float, default=0.5)
    parser.add_argument("--tb_top_k", type=int, default=5)
    parser.add_argument("--tb_selection_mode", type=str, default="top_k")
    parser.add_argument("--tb_summarization_length", type=int, default=100)
    parser.add_argument("--tb_summarization_model", type=str, default=None)
    parser.add_argument("--tb_embedding_model", type=str, default=None)
    parser.add_argument("--tb_cluster_embedding_model", type=str, default=None)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    run_case_study(args)


if __name__ == "__main__":
    main()