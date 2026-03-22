from pathlib import Path

from src.AINSTEIN.components.evaluation import Evaluation
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


def test_similarity_metrics_are_bounded(tmp_path: Path):
    evaluation = Evaluation(make_config(tmp_path))
    left = "robust multimodal transformer with retrieval augmentation"
    right = "retrieval augmented multimodal transformer for robust reasoning"

    precision, recall, f1 = evaluation._token_f1(left, right)

    assert 0.0 <= evaluation._jaccard_similarity(left, right) <= 1.0
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert 0.0 <= evaluation._keyword_overlap(left, right) <= 1.0
    assert 0.0 <= evaluation._length_ratio(left, right) <= 1.0
    assert 0.0 <= evaluation._sequence_similarity(left, right) <= 1.0


def test_identical_strings_have_high_overlap(tmp_path: Path):
    evaluation = Evaluation(make_config(tmp_path))
    text = "graph neural network with sparse message passing"
    precision, recall, f1 = evaluation._token_f1(text, text)

    assert evaluation._jaccard_similarity(text, text) == 1.0
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0
    assert evaluation._length_ratio(text, text) == 1.0
