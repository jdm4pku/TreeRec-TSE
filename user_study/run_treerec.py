"""
TreeRec for user study queries.
Loads pre-built gpt4o trees and runs tree-guided search + re-rank.
Connects via local proxy (localhost:4141).
"""

import os
import sys
import json
import time
import pickle

# os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["OPENAI_API_KEY"] = "sk-M2uzKmvMH7cODM6KObmBXkuyZpknX8MXPptFOeUJgADxGr6Q"
os.environ["OPENAI_BASE_URL"] = "https://api.chatanywhere.tech/v1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
TREEREC_DIR = os.path.join(ROOT_DIR, "TreeRec-code")

sys.path.insert(0, TREEREC_DIR)

from TreeRec.RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig

QUERY_DIR = os.path.join(BASE_DIR, "participant_query")
OUTPUT_DIR = os.path.join(BASE_DIR, "approach_output", "TreeRec")
DATA_DIR = os.path.join(TREEREC_DIR, "IntentRecBench", "data")
TREE_DIR = os.path.join(TREEREC_DIR, "output", "TreeRec", "gpt4o")

TOP_K = 5


def load_treerec(ecosystem: str) -> RetrievalAugmentation:
    tree_path = os.path.join(TREE_DIR, f"{ecosystem}-tree.pkl")
    if not os.path.exists(tree_path):
        raise FileNotFoundError(f"Tree not found: {tree_path}")

    config = RetrievalAugmentationConfig(
        rerank_model="gpt-4o-2024-05-13",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        summarization_model="gpt-4o-2024-05-13",
        tree_builder_type="cluster",
        use_rerank=True,
        tr_threshold=0.5,
        tr_top_k=40,
        tr_selection_mode="top_k",
        tb_max_tokens=100,
        tb_num_layers=5,
        tb_threshold=0.5,
        tb_top_k=5,
        tb_selection_mode="top_k",
        tb_summarization_length=100,
        tb_cluster_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    RA = RetrievalAugmentation(config=config, tree=tree_path)
    print(f"Loaded tree for {ecosystem} from {tree_path}")
    return RA


def get_desc_map(ecosystem: str) -> dict:
    with open(os.path.join(DATA_DIR, ecosystem, "candidate_artifacts.json")) as f:
        cands = json.load(f)
    return {c["name"]: c.get("description", "") for c in cands}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    eco_map = {"JavaScript": "js", "HuggingFace": "hf"}

    # Pre-load trees and description maps
    ra_cache = {}
    desc_cache = {}
    for eco in ["js", "hf"]:
        ra_cache[eco] = load_treerec(eco)
        desc_cache[eco] = get_desc_map(eco)

    # Process each participant
    for fname in sorted(os.listdir(QUERY_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(QUERY_DIR, fname)) as f:
            pdata = json.load(f)

        pid = pdata["participant_id"]
        eco = eco_map[pdata["ecosystem"]]
        RA = ra_cache[eco]
        desc_map = desc_cache[eco]

        print(f"\n{'='*50}")
        print(f"Processing {pid} ({pdata['ecosystem']}, {len(pdata['queries'])} queries)")
        print(f"{'='*50}")

        results = []
        for qi, q in enumerate(pdata["queries"], 1):
            query = q["your_query"]
            print(f"\n  Query {qi}/6: {query[:60]}...")

            t0 = time.time()
            top_k_names = RA.artifact_recommendation(query, top_k=TOP_K)
            elapsed = time.time() - t0

            top5 = []
            for rank, name in enumerate(top_k_names, 1):
                top5.append({
                    "rank": rank,
                    "artifact_name": name,
                    "description": desc_map.get(name, "")
                })

            results.append({
                "scenario_id": q["scenario_id"],
                "query": query,
                "top5_recommendations": top5
            })
            print(f"  Done in {elapsed:.1f}s. Top-1: {top_k_names[0] if top_k_names else 'N/A'}")

        output = {
            "participant_id": pid,
            "ecosystem": pdata["ecosystem"],
            "method": "TreeRec",
            "results": results
        }
        out_path = os.path.join(OUTPUT_DIR, f"{pid}_TreeRec.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {pid} results to {out_path}")


if __name__ == "__main__":
    main()
