"""Generate a small, realistic sample evaluation dataset for the dashboard demo.

Writes into web/sample/evaluation/ so the website looks populated out-of-the-box
(enabled with AINSTEIN_SAMPLE=1) without needing a real, costly batch run.

    python scripts/make_sample_data.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "web" / "sample" / "evaluation"

PAPERS = [
    ("iclr_0001", "Sparse Mixture-of-Experts Routing with Learned Token Dropping", "Oral",
     0.92, True, True, 0, "Sparse MoE with a learned router that drops low-salience tokens.", 0.41),
    ("iclr_0002", "Contrastive Pretraining for Long-Horizon Robotic Manipulation", "Oral",
     0.88, True, False, 1, "Contrastive objective aligning visual frames with action sequences.", 0.36),
    ("iclr_0003", "Diffusion Priors for Inverse Problems in Medical Imaging", "Spotlight",
     0.74, False, True, 1, "Plug-and-play diffusion prior with a data-consistency projection step.", 0.33),
    ("iclr_0004", "Provable Convergence of Decentralized Adam under Heterogeneity", "Spotlight",
     0.0, False, False, 0, "A bias-corrected gradient-tracking variant of decentralized Adam.", 0.18),
    ("iclr_0005", "Retrieval-Augmented Reasoning over Structured Knowledge Graphs", "Poster",
     0.81, True, False, 0, "Hop-wise retrieval that conditions decoding on subgraph evidence.", 0.39),
    ("iclr_0006", "Adaptive Token Merging for Efficient Vision Transformers", "Poster",
     0.0, False, True, 1, "Similarity-based token merging scheduled by layer-wise budgets.", 0.31),
]

SUMMARY_METRICS = [
    "success_rate_relaxed", "success_rate_strict", "rediscovery_relaxed", "rediscovery_strict",
    "novel_and_valid_relaxed", "novel_and_valid_strict", "judge_agreement",
    "token_jaccard", "token_f1", "keyword_overlap", "length_ratio", "sequence_similarity",
]


def _row(paper):
    pid, title, tier, _conf, redis, novel, _idx, solution, tf1 = paper
    solved = redis or novel
    return {
        "paper_id": pid, "title": title, "tier": tier, "row_index": PAPERS.index(paper),
        "method_name": "main_model",
        "problem_statement": f"Design a method that addresses the core challenge described in '{title}'.",
        "model_solution": solution,
        "success_rate_relaxed": solved, "success_rate_strict": solved and redis,
        "rediscovery_relaxed": redis, "rediscovery_strict": redis and tier == "Oral",
        "novel_and_valid_relaxed": novel, "novel_and_valid_strict": False,
        "judge_agreement": True,
        "judge_1_feasible_and_complete": solved, "judge_1_rediscovery": redis,
        "judge_1_novel_and_valid": novel,
        "judge_1_justification": "The approach is technically feasible and addresses the problem.",
        "judge_2_feasible_and_complete": solved, "judge_2_rediscovery": redis,
        "judge_2_novel_and_valid": novel,
        "judge_2_justification": "Consistent with the first judge; method is sound.",
        "token_jaccard": round(tf1 * 0.6, 4), "token_precision": round(tf1 * 1.05, 4),
        "token_recall": round(tf1 * 0.95, 4), "token_f1": tf1,
        "keyword_overlap": round(tf1 * 0.8, 4), "length_ratio": round(0.5 + tf1 * 0.5, 4),
        "sequence_similarity": round(tf1 * 1.1, 4),
        "justification": "Judge 1 and Judge 2 agree on the verdict.", "error": "",
    }


def _summarize(df, group_col=None):
    def agg(frame):
        out = {"num_papers": len(frame), "num_errors": int((frame["error"].astype(str) != "").sum())}
        for m in SUMMARY_METRICS:
            out[m] = round(float(frame[m].astype(float).mean()), 4)
        return out

    if group_col is None:
        return pd.DataFrame([agg(df)])
    rows = []
    for key, frame in df.groupby(group_col):
        row = {group_col: key}
        row.update(agg(frame))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame([_row(p) for p in PAPERS])
    results.to_csv(OUT / "evaluation_results.csv", index=False)
    _summarize(results).to_csv(OUT / "evaluation_summary.csv", index=False)
    _summarize(results, "tier").to_csv(OUT / "evaluation_tier_summary.csv", index=False)

    # Three simple baselines, deliberately weaker than the main model.
    baselines = []
    for name, sr in [("problem_restatement", 0.0), ("keyword_template", 0.0), ("abstract_copy", 0.17)]:
        baselines.append({
            "method_name": name, "num_papers": len(PAPERS), "num_errors": 0,
            "success_rate_relaxed": sr, "success_rate_strict": 0.0,
            "rediscovery_relaxed": sr, "rediscovery_strict": 0.0,
            "novel_and_valid_relaxed": 0.0, "novel_and_valid_strict": 0.0,
            "judge_agreement": 0.83,
            "token_jaccard": 0.15 if name != "abstract_copy" else 0.45,
            "token_f1": 0.2 if name != "abstract_copy" else 0.55,
            "keyword_overlap": 0.2, "length_ratio": 0.4, "sequence_similarity": 0.3,
        })
    pd.DataFrame(baselines).to_csv(OUT / "baseline_summary.csv", index=False)

    report = (
        "# AINSTEIN Experiment Report (sample)\n\n"
        f"- Papers evaluated: {len(PAPERS)}\n"
        "- Overall relaxed success: 50.0%\n"
        "- Overall strict success: 33.3%\n"
        "- Rediscovery (relaxed): 33.3%\n"
        "- Judge agreement: 66.7%\n\n"
        "This is bundled sample data for the dashboard demo (AINSTEIN_SAMPLE=1).\n"
    )
    (OUT / "experiment_report.md").write_text(report, encoding="utf-8")
    print(f"Sample data written to {OUT}")


if __name__ == "__main__":
    main()
