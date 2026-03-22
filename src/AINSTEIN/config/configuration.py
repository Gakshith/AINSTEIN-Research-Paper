from src.AINSTEIN.constants import CONFIG_FILE_PATH
from dataclasses import dataclass
from pathlib import Path
from src.AINSTEIN.entity.config_entity import DataIngestionConfig,DataValidationConfig,GeneralizerConfig,SolverConfig,InternalCritiqueConfig,ExternalCritiqueConfig,EvaluationConfig
from src.AINSTEIN.constants import CONFIG_FILE_PATH,SCHEMA_FILE_PATH
from src.AINSTEIN.utils.common import read_yaml,create_directories,get_size


class ConfigurationManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH,schema_filepath=SCHEMA_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.schema = read_yaml(schema_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self)->DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir = config.root_dir,
            source_URL = config.source_URL,
            venue_id=config.venue_id,
            local_data_file = config.local_data_file,
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        create_directories([config.root_dir])

        return DataValidationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            STATUS_FILE=config.STATUS_FILE,
            all_schema=schema
        )
    def get_generalizer_config(self) -> GeneralizerConfig:
        config = self.config.generalizer
        create_directories([config.root_dir])

        return GeneralizerConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            abstract_column=config.abstract_column,
            pdf_url_column=config.pdf_url_column,
            paper_id=config.paper_id,
            row_index=config.row_index,
            abstract_fallback_to_csv=config.abstract_fallback_to_csv,
            generalizer_model=config.generalizer_model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            output_file=config.output_file,
            reference_solution_file=config.reference_solution_file,
        )
    def get_solver_config(self) -> SolverConfig:
        config = self.config.solver
        create_directories([config.root_dir])
        return SolverConfig(
            root_dir= config.root_dir,
            data_path= config.data_path,
            solver_model= config.solver_model,
            max_tokens= config.max_tokens,
            temperature= config.temperature,
            output_file= config.output_file
        )
    def get_internal_critique_config(self):
        config = self.config.internal_critique
        create_directories([config.root_dir])
        return InternalCritiqueConfig(
            root_dir=config.root_dir,
            abstract_path = config.abstract_path,
            solution_path = config.solution_path,
            internal_critique_model=config.internal_critique_model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            internal_critique_threshold=config.internal_critique_threshold,
            output_file=config.output_file
        )
    def get_external_critique_config(self):
        config = self.config.external_critique
        create_directories([config.root_dir])
        return ExternalCritiqueConfig(
            root_dir=config.root_dir,
            abstract_path = config.abstract_path,
            solution_path = config.solution_path,
            external_critique_model = config.external_critique_model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            external_critique_threshold=config.external_critique_threshold,
            output_file=config.output_file,
            internal_critique_status = config.internal_critique_status
        )
    def get_evaluation_config(self):
        config = self.config.evaluation
        create_directories([config.root_dir])
        return EvaluationConfig(
            root_dir=config.root_dir,
            problem_statement_path=config.problem_statement_path,
            generated_solution_path=config.generated_solution_path,
            reference_solution_path=config.reference_solution_path,
            internal_status_path=config.internal_status_path,
            external_status_path=config.external_status_path,
            evaluation_model=config.evaluation_model,
            evaluation_model_secondary=config.evaluation_model_secondary,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            output_file=config.output_file,
            results_csv=config.results_csv,
            summary_csv=config.summary_csv,
            tier_summary_csv=config.tier_summary_csv,
            baseline_results_csv=config.baseline_results_csv,
            baseline_summary_csv=config.baseline_summary_csv,
            plots_dir=config.plots_dir,
            overall_table_md=config.overall_table_md,
            tier_table_md=config.tier_table_md,
            baseline_table_md=config.baseline_table_md,
            experiment_report_md=config.experiment_report_md,
        )
