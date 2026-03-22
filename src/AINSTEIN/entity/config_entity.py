from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    venue_id: str
    local_data_file: Path

from dataclasses import dataclass
@dataclass(frozen=True)
class DataValidationConfig:
    root_dir : Path
    data_path : Path
    STATUS_FILE: Path
    all_schema: dict

@dataclass(frozen=True)
class GeneralizerConfig:
    root_dir: Path
    data_path: Path
    abstract_column: str
    pdf_url_column: str
    paper_id: str | None
    row_index: int
    abstract_fallback_to_csv: bool
    generalizer_model: str
    max_tokens: int
    temperature: float
    output_file: Path
    reference_solution_file: Path

@dataclass(frozen=True)
class SolverConfig:
    root_dir: Path
    data_path: Path
    solver_model: str
    max_tokens: int
    temperature: float
    output_file: Path
@dataclass(frozen=True)
class InternalCritiqueConfig:
  root_dir: Path
  abstract_path:Path
  solution_path: Path
  internal_critique_model: str
  max_tokens: int
  temperature: int
  internal_critique_threshold: int
  output_file: Path

@dataclass(frozen=True)
class ExternalCritiqueConfig:
  root_dir: Path
  abstract_path:Path
  solution_path: Path
  external_critique_model: str
  max_tokens: int
  temperature: int
  external_critique_threshold: int
  output_file: Path
  internal_critique_status: Path

@dataclass(frozen=True)
class EvaluationConfig:
  root_dir: Path
  problem_statement_path: Path
  generated_solution_path: Path
  reference_solution_path: Path
  internal_status_path: Path
  external_status_path: Path
  evaluation_model: str
  evaluation_model_secondary: str
  max_tokens: int
  temperature: float
  output_file: Path
  results_csv: Path
  summary_csv: Path
  tier_summary_csv: Path
  baseline_results_csv: Path
  baseline_summary_csv: Path
  plots_dir: Path
  overall_table_md: Path
  tier_table_md: Path
  baseline_table_md: Path
  experiment_report_md: Path
