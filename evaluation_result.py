from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.AINSTEIN import logger
from src.AINSTEIN.components.baselines import BaselineGenerator
from src.AINSTEIN.components.evaluation import Evaluation
from src.AINSTEIN.components.external_critique import ExternalCritique
from src.AINSTEIN.components.generalizer import Generalizer
from src.AINSTEIN.components.internal_critique import InternalCritique
from src.AINSTEIN.components.reporting import ExperimentReporter
from src.AINSTEIN.components.solver import Solver
from src.AINSTEIN.config.configuration import ConfigurationManager
from src.AINSTEIN.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.AINSTEIN.pipeline.stage_02_data_validation import DataValidationTrainingPipeline


class BatchEvaluationController:
    def __init__(self, max_internal_attempts: int = 3, max_external_attempts: int = 3):
        self.max_internal_attempts = max_internal_attempts
        self.max_external_attempts = max_external_attempts
        self.baseline_names = [
            "problem_restatement",
            "keyword_template",
            "abstract_copy",
        ]

    def _run_stage(self, stage_name: str, callable_obj):
        try:
            logger.info(f">>>>>> stage {stage_name} started <<<<<<")
            result = callable_obj()
            logger.info(f">>>>>> stage {stage_name} completed <<<<<<")
            return result
        except Exception as e:
            logger.exception(e)
            raise e

    def _read_status(self, file_path: Path) -> bool:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("Status:"):
                        return line.split(":", 1)[1].strip().lower() == "true"
        except FileNotFoundError:
            logger.warning(f"Status file not found: {file_path}")
        return False

    def _write_text(self, path: Path, content: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")

    def _clear_iteration_artifacts(self, config_manager: ConfigurationManager):
        paths_to_clear = [
            config_manager.get_generalizer_config().output_file,
            config_manager.get_generalizer_config().reference_solution_file,
            config_manager.get_solver_config().output_file,
            config_manager.get_internal_critique_config().output_file,
            config_manager.get_external_critique_config().output_file,
            config_manager.get_evaluation_config().output_file,
        ]

        for artifact_path in paths_to_clear:
            path = Path(artifact_path)
            if path.exists():
                path.unlink()

    def _write_summary_reports(self, results_df: pd.DataFrame, config_manager: ConfigurationManager):
        evaluation_config = config_manager.get_evaluation_config()
        summary_path = Path(evaluation_config.summary_csv)
        tier_summary_path = Path(evaluation_config.tier_summary_csv)
        baseline_summary_path = Path(evaluation_config.baseline_summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        summary_columns = [
            "success_rate_relaxed",
            "success_rate_strict",
            "rediscovery_relaxed",
            "rediscovery_strict",
            "novel_and_valid_relaxed",
            "novel_and_valid_strict",
            "judge_agreement",
            "token_jaccard",
            "token_f1",
            "keyword_overlap",
            "length_ratio",
            "sequence_similarity",
        ]

        main_results = results_df[results_df["method_name"] == "main_model"].copy()
        baseline_results = results_df[results_df["method_name"] != "main_model"].copy()

        overall_summary = {
            "num_papers": len(main_results),
            "num_errors": int(main_results["error"].astype(str).ne("").sum()),
        }
        for column in summary_columns:
            overall_summary[column] = float(main_results[column].fillna(False).astype(float).mean())

        pd.DataFrame([overall_summary]).to_csv(summary_path, index=False)

        tier_summary = (
            main_results.assign(has_error=main_results["error"].astype(str).ne(""))
            .groupby("tier", dropna=False)
            .agg(
                num_papers=("paper_id", "count"),
                num_errors=("has_error", "sum"),
                success_rate_relaxed=("success_rate_relaxed", "mean"),
                success_rate_strict=("success_rate_strict", "mean"),
                rediscovery_relaxed=("rediscovery_relaxed", "mean"),
                rediscovery_strict=("rediscovery_strict", "mean"),
                novel_and_valid_relaxed=("novel_and_valid_relaxed", "mean"),
                novel_and_valid_strict=("novel_and_valid_strict", "mean"),
                judge_agreement=("judge_agreement", "mean"),
                token_jaccard=("token_jaccard", "mean"),
                token_f1=("token_f1", "mean"),
                keyword_overlap=("keyword_overlap", "mean"),
                length_ratio=("length_ratio", "mean"),
                sequence_similarity=("sequence_similarity", "mean"),
            )
            .reset_index()
        )
        tier_summary.to_csv(tier_summary_path, index=False)

        if not baseline_results.empty:
            baseline_summary = (
                baseline_results.assign(has_error=baseline_results["error"].astype(str).ne(""))
                .groupby("method_name", dropna=False)
                .agg(
                    num_papers=("paper_id", "count"),
                    num_errors=("has_error", "sum"),
                    success_rate_relaxed=("success_rate_relaxed", "mean"),
                    success_rate_strict=("success_rate_strict", "mean"),
                    rediscovery_relaxed=("rediscovery_relaxed", "mean"),
                    rediscovery_strict=("rediscovery_strict", "mean"),
                    novel_and_valid_relaxed=("novel_and_valid_relaxed", "mean"),
                    novel_and_valid_strict=("novel_and_valid_strict", "mean"),
                    judge_agreement=("judge_agreement", "mean"),
                    token_jaccard=("token_jaccard", "mean"),
                    token_f1=("token_f1", "mean"),
                    keyword_overlap=("keyword_overlap", "mean"),
                    length_ratio=("length_ratio", "mean"),
                    sequence_similarity=("sequence_similarity", "mean"),
                )
                .reset_index()
            )
            baseline_summary.to_csv(baseline_summary_path, index=False)
        else:
            pd.DataFrame(columns=["method_name"]).to_csv(baseline_summary_path, index=False)

        logger.info(f"Overall evaluation summary saved at: {summary_path}")
        logger.info(f"Tier evaluation summary saved at: {tier_summary_path}")
        logger.info(f"Baseline summary saved at: {baseline_summary_path}")

    def _build_result_row(
        self,
        row: pd.Series,
        row_index: int,
        method_name: str,
        problem_statement: str,
        model_solution: str,
        evaluation_result: dict,
        error: str = "",
    ) -> dict:
        return {
            "paper_id": row.get("paper_id", ""),
            "title": row.get("title", ""),
            "tier": row.get("tier", ""),
            "row_index": row_index,
            "method_name": method_name,
            "problem_statement": problem_statement,
            "model_solution": model_solution or "",
            "success_rate_relaxed": evaluation_result.get("success_rate_relaxed", False),
            "success_rate_strict": evaluation_result.get("success_rate_strict", False),
            "rediscovery_relaxed": evaluation_result.get("rediscovery_relaxed", False),
            "rediscovery_strict": evaluation_result.get("rediscovery_strict", False),
            "novel_and_valid_relaxed": evaluation_result.get("novel_and_valid_relaxed", False),
            "novel_and_valid_strict": evaluation_result.get("novel_and_valid_strict", False),
            "judge_agreement": evaluation_result.get("judge_agreement", False),
            "judge_1_feasible_and_complete": evaluation_result.get("judge_1_feasible_and_complete", False),
            "judge_1_rediscovery": evaluation_result.get("judge_1_rediscovery", False),
            "judge_1_novel_and_valid": evaluation_result.get("judge_1_novel_and_valid", False),
            "judge_1_justification": evaluation_result.get("judge_1_justification", ""),
            "judge_2_feasible_and_complete": evaluation_result.get("judge_2_feasible_and_complete", False),
            "judge_2_rediscovery": evaluation_result.get("judge_2_rediscovery", False),
            "judge_2_novel_and_valid": evaluation_result.get("judge_2_novel_and_valid", False),
            "judge_2_justification": evaluation_result.get("judge_2_justification", ""),
            "token_jaccard": evaluation_result.get("token_jaccard", 0.0),
            "token_precision": evaluation_result.get("token_precision", 0.0),
            "token_recall": evaluation_result.get("token_recall", 0.0),
            "token_f1": evaluation_result.get("token_f1", 0.0),
            "keyword_overlap": evaluation_result.get("keyword_overlap", 0.0),
            "length_ratio": evaluation_result.get("length_ratio", 0.0),
            "sequence_similarity": evaluation_result.get("sequence_similarity", 0.0),
            "justification": evaluation_result.get("justification", ""),
            "error": error,
        }

    def _evaluate_baselines(
        self,
        row: pd.Series,
        row_index: int,
        config_manager: ConfigurationManager,
        problem_statement: str,
        abstract_text: str,
    ) -> list[dict]:
        baseline_generator = BaselineGenerator()
        evaluation_config = config_manager.get_evaluation_config()
        solver_config = config_manager.get_solver_config()
        generalizer_config = config_manager.get_generalizer_config()

        reference_solution = Path(generalizer_config.reference_solution_file).read_text(encoding="utf-8").strip()
        baseline_rows = []
        for baseline_name in self.baseline_names:
            try:
                baseline_solution = baseline_generator.generate(
                    baseline_name,
                    problem_statement,
                    reference_solution,
                    abstract_text,
                )
                self._write_text(Path(solver_config.output_file), baseline_solution)
                evaluation_result = Evaluation(evaluation_config).evaluate_solution(critique_override=True)
                baseline_rows.append(
                    self._build_result_row(
                        row,
                        row_index,
                        baseline_name,
                        problem_statement,
                        baseline_solution,
                        evaluation_result,
                    )
                )
            except Exception as exc:
                logger.exception(exc)
                baseline_rows.append(
                    self._build_result_row(
                        row,
                        row_index,
                        baseline_name,
                        problem_statement,
                        "",
                        {},
                        error=str(exc),
                    )
                )
        return baseline_rows

    def _run_single_paper(self, row_index: int, row: pd.Series, config_manager: ConfigurationManager) -> list[dict]:
        self._clear_iteration_artifacts(config_manager)

        generalizer_config = replace(
            config_manager.get_generalizer_config(),
            row_index=row_index,
            paper_id=None,
        )
        generalizer = Generalizer(generalizer_config)
        abstract_text = generalizer.load_abstract()
        problem_statement = generalizer.generalization_agent(abstract_text).generalized_research_abstract

        solver_config = config_manager.get_solver_config()
        internal_config = config_manager.get_internal_critique_config()
        external_config = config_manager.get_external_critique_config()
        evaluation_config = config_manager.get_evaluation_config()

        model_solution = ""
        for external_attempt in range(1, self.max_external_attempts + 1):
            logger.info(f"====== External Attempt {external_attempt}/{self.max_external_attempts} ======")

            for internal_attempt in range(1, self.max_internal_attempts + 1):
                logger.info(f"------ Internal Attempt {internal_attempt}/{self.max_internal_attempts} ------")

                solver = Solver(solver_config)
                model_solution = solver.get_solution().research_solution

                internal_critique = InternalCritique(internal_config)
                internal_critique.get_internal_critique_score()
                if self._read_status(Path(internal_config.output_file)):
                    logger.info("Internal critique passed.")
                    break

                logger.info("Internal critique failed. Sending solution back to Solver.")
            else:
                logger.info("Internal critique did not pass within max attempts. Moving to next external cycle.")
                continue

            external_critique = ExternalCritique(external_config)
            external_critique.get_external_critique_score()
            if self._read_status(Path(external_config.output_file)):
                logger.info("External critique passed. Final solution accepted.")
                break

            logger.info("External critique failed. Sending solution back to Solver.")

        evaluation = Evaluation(evaluation_config)
        evaluation_result = evaluation.evaluate_solution()
        main_row = self._build_result_row(
            row,
            row_index,
            "main_model",
            problem_statement,
            model_solution,
            evaluation_result,
        )

        baseline_rows = self._evaluate_baselines(
            row,
            row_index,
            config_manager,
            problem_statement,
            abstract_text,
        )
        return [main_row] + baseline_rows

    def run(self):
        self._run_stage("Data Ingestion stage", lambda: DataIngestionTrainingPipeline().main())
        validation_passed = self._run_stage(
            "Data Validation stage",
            lambda: DataValidationTrainingPipeline().main(),
        )
        if not validation_passed:
            raise ValueError("Data validation failed. Aborting batch evaluation.")

        config_manager = ConfigurationManager()
        dataset = pd.read_csv(config_manager.get_generalizer_config().data_path)
        evaluation_config = config_manager.get_evaluation_config()
        results_path = Path(evaluation_config.results_csv)
        baseline_results_path = Path(evaluation_config.baseline_results_csv)
        results_path.parent.mkdir(parents=True, exist_ok=True)

        all_results = []
        for row_index, row in dataset.iterrows():
            logger.info(f"========== Evaluating paper {row_index + 1}/{len(dataset)} ==========")
            try:
                all_results.extend(self._run_single_paper(row_index, row, config_manager))
            except Exception as exc:
                logger.exception(exc)
                all_results.append(
                    self._build_result_row(
                        row,
                        row_index,
                        "main_model",
                        "",
                        "",
                        {},
                        error=str(exc),
                    )
                )

        results_df = pd.DataFrame(all_results)
        main_results_df = results_df[results_df["method_name"] == "main_model"].copy()
        baseline_results_df = results_df[results_df["method_name"] != "main_model"].copy()

        main_results_df.to_csv(results_path, index=False)
        baseline_results_df.to_csv(baseline_results_path, index=False)
        self._write_summary_reports(results_df, config_manager)
        reporter = ExperimentReporter(evaluation_config)
        reporter.generate_tables_and_report()
        reporter.generate_plots()

        logger.info(f"Batch evaluation results saved at: {results_path}")
        logger.info(f"Baseline evaluation results saved at: {baseline_results_path}")
        return results_path


if __name__ == "__main__":
    controller = BatchEvaluationController(max_internal_attempts=3, max_external_attempts=3)
    results_path = controller.run()
    logger.info(f"Evaluation results CSV: {results_path}")
