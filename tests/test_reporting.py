from pathlib import Path

import pandas as pd

from src.AINSTEIN.components.reporting import ExperimentReporter
from src.AINSTEIN.entity.config_entity import EvaluationConfig


def make_config(tmp_path: Path) -> EvaluationConfig:
    return EvaluationConfig(
        root_dir=tmp_path,
        problem_statement_path=tmp_path / "problem.txt",
        generated_solution_path=tmp_path / "generated.txt",
        reference_solution_path=tmp_path / "reference.txt",
        internal_status_path=tmp_path / "internal.txt",
        external_status_path=tmp_path / "external.txt",
        evaluation_model="dummy",
        evaluation_model_secondary="dummy",
        max_tokens=128,
        temperature=0.0,
        output_file=tmp_path / "evaluation.txt",
        results_csv=tmp_path / "results.csv",
        summary_csv=tmp_path / "summary.csv",
        tier_summary_csv=tmp_path / "tier_summary.csv",
        baseline_results_csv=tmp_path / "baseline_results.csv",
        baseline_summary_csv=tmp_path / "baseline_summary.csv",
        plots_dir=tmp_path / "plots",
    )


def test_reporter_generates_plot_files(tmp_path: Path):
    config = make_config(tmp_path)
    pd.DataFrame(
        [
            {"tier": "Oral", "judge_agreement": 0.8},
            {"tier": "Poster", "judge_agreement": 0.5},
        ]
    ).to_csv(config.results_csv, index=False)
    pd.DataFrame(
        [
            {
                "tier": "Oral",
                "success_rate_relaxed": 0.8,
                "success_rate_strict": 0.6,
                "rediscovery_relaxed": 0.4,
                "novel_and_valid_relaxed": 0.3,
            }
        ]
    ).to_csv(config.tier_summary_csv, index=False)
    pd.DataFrame(
        [
            {
                "method_name": "problem_restatement",
                "success_rate_relaxed": 0.1,
                "success_rate_strict": 0.0,
                "rediscovery_relaxed": 0.0,
                "novel_and_valid_relaxed": 0.1,
            }
        ]
    ).to_csv(config.baseline_summary_csv, index=False)

    ExperimentReporter(config).generate_plots()

    assert (config.plots_dir / "tier_metrics.png").exists()
    assert (config.plots_dir / "baseline_comparison.png").exists()
    assert (config.plots_dir / "judge_agreement.png").exists()
