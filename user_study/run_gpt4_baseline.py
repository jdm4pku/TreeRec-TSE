"""
GPT-4 baseline for user study queries.
Uses the two-stage strategy from IntentRecBench/src/baselines/llm.py,
adapted for user study participant queries.
Connects via local proxy (localhost:4141) as in test-sdk.py.
"""

import os
import sys
import json
import time
import re
from typing import List, Dict

os.environ["no_proxy"] = "localhost,127.0.0.1"

from openai import OpenAI

BASE_URL = "https://api.chatanywhere.tech/v1"
MODEL_NAME = "gpt-4"
TOP_K = 5
BATCH_SIZE = 20
FILTER_TOP_PERCENT = 0.1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "TreeRec-code", "IntentRecBench", "data")
QUERY_DIR = os.path.join(BASE_DIR, "participant_query")
OUTPUT_DIR = os.path.join(BASE_DIR, "approach_output", "GPT4")

client = OpenAI(base_url=BASE_URL, api_key="sk-M2uzKmvMH7cODM6KObmBXkuyZpknX8MXPptFOeUJgADxGr6Q")


def call_llm(prompt: str, system_content: str = None, max_tokens: int = 2000) -> str:
    system_content = system_content or "You are a professional artifact recommendation assistant."
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=max_tokens
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 4:
                wait = min(5 * (attempt + 1), 30)
                print(f"  Retry {attempt+1}/5: {e}, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


def build_batch_scoring_prompt(intent: str, artifacts: List[Dict], ecosystem: str) -> str:
    eco_names = {"hf": "Hugging Face", "js": "npm/JavaScript"}
    eco_name = eco_names.get(ecosystem, ecosystem)

    parts = []
    for idx, a in enumerate(artifacts, 1):
        desc = a.get("description", "No description")
        if ecosystem == "hf" and len(desc) > 500:
            desc = desc[:500] + " ... (truncated)"
        parts.append(f"{idx}. Name: {a['name']}\n   Description: {desc}")

    return f"""You are an expert in {eco_name} ecosystem artifact recommendation. Please evaluate the semantic relevance between each of the following artifacts and the user intent.

User Intent:
{intent}

Artifact List:
{chr(10).join(parts)}

Please provide a semantic relevance score (0-100 integer) for each artifact, where 100 indicates a perfect match and 0 indicates complete irrelevance.
Return the scores in the following format (one score per line, in the same order as the artifacts):
score1
score2
...

Return only the numbers, one per line, without any additional text."""


def build_recommend_prompt(intent: str, candidates: List[Dict], ecosystem: str, top_k: int) -> str:
    eco_names = {"hf": "Hugging Face", "js": "npm/JavaScript"}
    eco_name = eco_names.get(ecosystem, ecosystem)

    parts = []
    for idx, a in enumerate(candidates, 1):
        desc = a.get("description", "No description")
        if ecosystem == "hf" and len(desc) > 1000:
            desc = desc[:1000] + " ... (truncated)"
        parts.append(f"{idx}. Name: {a['name']}\n   Description: {desc}")

    return f"""You are an expert in {eco_name} ecosystem artifact recommendation. Based on the user's intent, recommend the most relevant artifacts from the given candidate list.

User Intent:
{intent}

Candidate Artifact List:
{chr(10).join(parts)}

Please select the top {top_k} most relevant artifacts from the above candidate list based on the user intent, ranked from highest to lowest relevance.
Return only the artifact names (the Name field), one per line, without any numbering, prefixes, or additional text.

Please directly output the recommended artifact names:"""


def score_batch(intent: str, artifacts: List[Dict], ecosystem: str) -> List[float]:
    prompt = build_batch_scoring_prompt(intent, artifacts, ecosystem)
    try:
        content = call_llm(prompt,
            system_content="You are a professional artifact semantic relevance evaluation assistant.",
            max_tokens=max(200, len(artifacts) * 50))
        scores = []
        for line in content.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            nums = re.findall(r'\d+', line)
            if nums:
                scores.append(max(0, min(100, float(nums[0]))))
        while len(scores) < len(artifacts):
            scores.append(50.0)
        return scores[:len(artifacts)]
    except Exception as e:
        print(f"  Batch scoring error: {e}")
        return [50.0] * len(artifacts)


def get_recommendations(intent: str, candidates: List[Dict], ecosystem: str) -> List[Dict]:
    """Two-stage: score all candidates, filter top 10%, then recommend top-5."""
    total = len(candidates)

    # Stage 1: Scoring
    scored = []
    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    for bi in range(num_batches):
        batch = candidates[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
        batch_scores = score_batch(intent, batch, ecosystem)
        for s, a in zip(batch_scores, batch):
            scored.append((s, a))
        if bi < num_batches - 1:
            time.sleep(0.1)
        if (bi + 1) % 50 == 0:
            print(f"    Scored {(bi+1)*BATCH_SIZE}/{total}")

    scored.sort(key=lambda x: x[0], reverse=True)
    top_n = max(1, int(total * FILTER_TOP_PERCENT))
    filtered = [a for _, a in scored[:top_n]]

    # Stage 2: Recommend
    candidate_names = [a["name"] for a in filtered]
    prompt = build_recommend_prompt(intent, filtered, ecosystem, TOP_K)
    try:
        content = call_llm(prompt)
        rec_names = []
        for line in content.split('\n'):
            line = line.strip().lstrip('0123456789.-*()[] ').strip()
            if line and line in candidate_names:
                rec_names.append(line)
        if len(rec_names) < TOP_K:
            remaining = [n for n in candidate_names if n not in rec_names]
            rec_names.extend(remaining[:TOP_K - len(rec_names)])
        rec_names = rec_names[:TOP_K]
    except Exception as e:
        print(f"  Recommend error: {e}")
        rec_names = candidate_names[:TOP_K]

    # Build output with descriptions
    desc_map = {a["name"]: a.get("description", "") for a in candidates}
    results = []
    for rank, name in enumerate(rec_names, 1):
        results.append({
            "rank": rank,
            "artifact_name": name,
            "description": desc_map.get(name, "")
        })
    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    eco_map = {"JavaScript": "js", "HuggingFace": "hf"}

    # Load candidates
    cand_cache = {}
    for eco in ["js", "hf"]:
        with open(os.path.join(DATA_DIR, eco, "candidate_artifacts.json")) as f:
            cand_cache[eco] = json.load(f)
        print(f"Loaded {len(cand_cache[eco])} candidates for {eco}")

    # Process each participant
    for fname in sorted(os.listdir(QUERY_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(QUERY_DIR, fname)) as f:
            pdata = json.load(f)

        pid = pdata["participant_id"]
        eco = eco_map[pdata["ecosystem"]]
        candidates = cand_cache[eco]

        print(f"\n{'='*50}")
        print(f"Processing {pid} ({pdata['ecosystem']}, {len(pdata['queries'])} queries)")
        print(f"{'='*50}")

        results = []
        for qi, q in enumerate(pdata["queries"], 1):
            query = q["your_query"]
            print(f"\n  Query {qi}/6: {query[:60]}...")

            t0 = time.time()
            top5 = get_recommendations(query, candidates, eco)
            elapsed = time.time() - t0

            results.append({
                "scenario_id": q["scenario_id"],
                "query": query,
                "top5_recommendations": top5
            })
            print(f"  Done in {elapsed:.1f}s. Top-1: {top5[0]['artifact_name']}")

        output = {
            "participant_id": pid,
            "ecosystem": pdata["ecosystem"],
            "method": "GPT-4",
            "results": results
        }
        out_path = os.path.join(OUTPUT_DIR, f"{pid}_GPT4.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {pid} results to {out_path}")


if __name__ == "__main__":
    main()
