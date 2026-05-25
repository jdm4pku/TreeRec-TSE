"""
Generate blind evaluation forms for each participant.
Each query's 3 methods (BM25, GPT-4, TreeRec) are randomly assigned to A/B/C.
"""

import os
import json
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "approach_output")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation_forms")
os.makedirs(EVAL_DIR, exist_ok=True)

METHODS = ["BM25", "GPT4", "TreeRec"]
METHOD_DIRS = {"BM25": "BM25", "GPT4": "GPT4", "TreeRec": "TreeRec"}
METHOD_SUFFIXES = {"BM25": "BM25", "GPT4": "GPT4", "TreeRec": "TreeRec"}

random.seed(42)

participants = [f"P{i}" for i in range(1, 13)]

for pid in participants:
    results_by_method = {}
    for method in METHODS:
        path = os.path.join(OUTPUT_DIR, METHOD_DIRS[method], f"{pid}_{METHOD_SUFFIXES[method]}.json")
        with open(path) as f:
            data = json.load(f)
        for r in data["results"]:
            sid = r["scenario_id"]
            if sid not in results_by_method:
                results_by_method[sid] = {}
            results_by_method[sid][method] = r["top5_recommendations"]

    with open(os.path.join(BASE_DIR, "participant_query", f"{pid}_query.json")) as f:
        pdata = json.load(f)

    queries = []
    method_assignment = {}

    for q in pdata["queries"]:
        sid = q["scenario_id"]
        shuffled = METHODS[:]
        random.shuffle(shuffled)
        labels = {shuffled[0]: "A", shuffled[1]: "B", shuffled[2]: "C"}
        method_assignment[sid] = {v: k for k, v in labels.items()}

        method_results = {}
        for method in METHODS:
            label = labels[method]
            recs = []
            for item in results_by_method[sid][method][:3]:
                recs.append({
                    "rank": item["rank"],
                    "artifact_name": item["artifact_name"],
                    "description": item.get("description", "")
                })
            method_results[label] = recs

        queries.append({
            "scenario_id": sid,
            "query": q["your_query"],
            "method_A": method_results["A"],
            "method_B": method_results["B"],
            "method_C": method_results["C"],
            "evaluation": {
                "method_A": {"relevance": None, "usefulness": None},
                "method_B": {"relevance": None, "usefulness": None},
                "method_C": {"relevance": None, "usefulness": None},
                "overall_ranking": {"best": None, "neutral": None, "worst": None}
            }
        })

    eval_form = {
        "participant_id": pid,
        "ecosystem": pdata["ecosystem"],
        "instructions": {
            "scoring": "For each method (A/B/C), rate the top-5 recommendations as a whole on: Relevance (1-5, how well do these recommendations match your intent?) and Usefulness (1-5, how useful would these recommendations be for solving your task?)",
            "ranking": "Rank the three methods: assign Best, Neutral, Worst to A/B/C respectively. Each label must be used exactly once.",
            "scale": "1=Very Poor, 2=Poor, 3=Average, 4=Good, 5=Excellent"
        },
        "queries": queries
    }

    # Save eval form (participant-facing)
    with open(os.path.join(EVAL_DIR, f"{pid}_eval.json"), "w", encoding="utf-8") as f:
        json.dump(eval_form, f, indent=2, ensure_ascii=False)

    # Save mapping (researcher-only)
    mapping = {"participant_id": pid, "assignments": method_assignment}
    with open(os.path.join(EVAL_DIR, f"{pid}_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    # Generate markdown evaluation form
    md_lines = [
        f"# Evaluation Form - {pid}",
        "",
        f"**Ecosystem:** {pdata['ecosystem']}",
        "",
        "## Instructions",
        "",
        "For each query below, three recommendation methods (A, B, C) each provide a top-3 list of recommended artifacts. Please:",
        "",
        "1. **Score** each method's recommendations as a whole on two dimensions (1-5 scale):",
        "   - **Relevance**: How well do these recommendations match your intent? (1=Very Poor, 5=Excellent)",
        "   - **Usefulness**: How useful would these recommendations be for solving your task? (1=Very Poor, 5=Excellent)",
        "2. **Rank** the three methods by assigning **Best**, **Neutral**, and **Worst** (each label used exactly once).",
        "",
        "---",
        "",
    ]

    for qi, q in enumerate(queries, 1):
        md_lines.append(f"## Query {qi} ({q['scenario_id']})")
        md_lines.append("")
        md_lines.append(f"> **Your query:** {q['query']}")
        md_lines.append("")

        for label in ["A", "B", "C"]:
            md_lines.append(f"### Method {label}")
            md_lines.append("")
            md_lines.append("| Rank | Artifact | Description |")
            md_lines.append("|------|----------|-------------|")
            for item in q[f"method_{label}"]:
                desc = item['description'].replace('|', '\\|').replace('\n', ' ')
                if len(desc) > 120:
                    desc = desc[:120] + "..."
                md_lines.append(f"| {item['rank']} | **{item['artifact_name']}** | {desc} |")
            md_lines.append("")

        md_lines.append("### Your Evaluation")
        md_lines.append("")
        md_lines.append("| Method | Relevance (1-5) | Usefulness (1-5) |")
        md_lines.append("|--------|-----------------|------------------|")
        md_lines.append("| A      |                 |                  |")
        md_lines.append("| B      |                 |                  |")
        md_lines.append("| C      |                 |                  |")
        md_lines.append("")
        md_lines.append("**Overall Ranking:**")
        md_lines.append("")
        md_lines.append("- Best: ____")
        md_lines.append("- Neutral: ____")
        md_lines.append("- Worst: ____")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    with open(os.path.join(EVAL_DIR, f"{pid}_eval.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Generated {pid} eval form, mapping, and markdown")

print(f"\nAll evaluation forms saved to {EVAL_DIR}")
