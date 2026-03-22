from pathlib import Path

import pandas as pd

from src.AINSTEIN import logger
from src.AINSTEIN.entity.config_entity import EvaluationConfig


class ExperimentReporter:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _load_csv(self, path: Path) -> pd.DataFrame:
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()

    def _write_markdown_table(self, df: pd.DataFrame, output_path: Path, title: str):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            if df.empty:
                f.write("No data available.\n")
            else:
                f.write(df.to_markdown(index=False))
                f.write("\n")
        logger.info(f"Saved markdown table at: {output_path}")

    def generate_tables_and_report(self):
        results_df = self._load_csv(Path(self.config.results_csv))
        summary_df = self._load_csv(Path(self.config.summary_csv))
        tier_summary_df = self._load_csv(Path(self.config.tier_summary_csv))
        baseline_summary_df = self._load_csv(Path(self.config.baseline_summary_csv))

        self._write_markdown_table(
            summary_df,
            Path(self.config.overall_table_md),
            "Overall Results Table",
        )
        self._write_markdown_table(
            tier_summary_df,
            Path(self.config.tier_table_md),
            "Tier Results Table",
        )
        self._write_markdown_table(
            baseline_summary_df,
            Path(self.config.baseline_table_md),
            "Baseline Results Table",
        )

        report_path = Path(self.config.experiment_report_md)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        num_papers = len(results_df) if not results_df.empty else 0
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Experiment Report\n\n")
            f.write(f"- Number of evaluated papers: {num_papers}\n")
            if not summary_df.empty:
                summary_row = summary_df.iloc[0]
                f.write(f"- Relaxed success rate: {summary_row.get('success_rate_relaxed', 0.0):.3f}\n")
                f.write(f"- Strict success rate: {summary_row.get('success_rate_strict', 0.0):.3f}\n")
                f.write(f"- Relaxed rediscovery: {summary_row.get('rediscovery_relaxed', 0.0):.3f}\n")
                f.write(f"- Relaxed novel & valid: {summary_row.get('novel_and_valid_relaxed', 0.0):.3f}\n")
                f.write(f"- Judge agreement: {summary_row.get('judge_agreement', 0.0):.3f}\n")
                f.write(f"- Mean token F1: {summary_row.get('token_f1', 0.0):.3f}\n")
            f.write("\n## Output Files\n\n")
            f.write(f"- Results CSV: `{self.config.results_csv}`\n")
            f.write(f"- Baseline Results CSV: `{self.config.baseline_results_csv}`\n")
            f.write(f"- Overall Summary CSV: `{self.config.summary_csv}`\n")
            f.write(f"- Tier Summary CSV: `{self.config.tier_summary_csv}`\n")
            f.write(f"- Baseline Summary CSV: `{self.config.baseline_summary_csv}`\n")
            f.write(f"- Overall Table: `{self.config.overall_table_md}`\n")
            f.write(f"- Tier Table: `{self.config.tier_table_md}`\n")
            f.write(f"- Baseline Table: `{self.config.baseline_table_md}`\n")
            f.write(f"- Plots Directory: `{self.config.plots_dir}`\n")
        logger.info(f"Saved experiment report at: {report_path}")

    def _save_bar_plot(self, df: pd.DataFrame, x_col: str, y_cols: list[str], output_path: Path, title: str):
        import matplotlib.pyplot as plt

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ax = df.set_index(x_col)[y_cols].plot(kind="bar", figsize=(10, 6))
        ax.set_title(title)
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1)
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved plot at: {output_path}")

    def generate_plots(self):
        results_path = Path(self.config.results_csv)
        summary_path = Path(self.config.tier_summary_csv)
        baseline_summary_path = Path(self.config.baseline_summary_csv)
        plots_dir = Path(self.config.plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

        try:
            import matplotlib.pyplot  # noqa: F401
        except ModuleNotFoundError:
            logger.warning("matplotlib is not installed. Skipping plot generation.")
            return

        if summary_path.exists():
            tier_df = pd.read_csv(summary_path)
            if not tier_df.empty:
                self._save_bar_plot(
                    tier_df,
                    "tier",
                    [
                        "success_rate_relaxed",
                        "success_rate_strict",
                        "rediscovery_relaxed",
                        "novel_and_valid_relaxed",
                    ],
                    plots_dir / "tier_metrics.png",
                    "Evaluation Metrics by Tier",
                )

        if baseline_summary_path.exists():
            baseline_df = pd.read_csv(baseline_summary_path)
            if not baseline_df.empty:
                self._save_bar_plot(
                    baseline_df,
                    "method_name",
                    [
                        "success_rate_relaxed",
                        "success_rate_strict",
                        "rediscovery_relaxed",
                        "novel_and_valid_relaxed",
                    ],
                    plots_dir / "baseline_comparison.png",
                    "Baseline Comparison",
                )

        if results_path.exists():
            results_df = pd.read_csv(results_path)
            if not results_df.empty:
                agreement_df = (
                    results_df.groupby("tier", dropna=False)["judge_agreement"]
                    .mean()
                    .reset_index()
                )
                self._save_bar_plot(
                    agreement_df,
                    "tier",
                    ["judge_agreement"],
                    plots_dir / "judge_agreement.png",
                    "Judge Agreement by Tier",
                )
