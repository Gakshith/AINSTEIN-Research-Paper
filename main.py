# from src.AINSTEIN import logger
# from src.AINSTEIN.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
#
# from src.AINSTEIN.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
# from src.AINSTEIN.pipeline.stage_03_generalizer import GeneralizerTrainingPipeline
# from src.AINSTEIN.pipeline.stage_04_solver import SolutionTrainingPipeline
# from src.AINSTEIN.pipeline.stage_05_internal_critique import InternalCritiqueTrainingPipeline
# from src.AINSTEIN.pipeline.stage_06_external_critique import ExternalCritiqueTrainingPipeline
#
#
# STAGE_NAME = "Data Ingestion stage"
# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#    data_ingestion = DataIngestionTrainingPipeline()
#    data_ingestion.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e
#
# STAGE_NAME = "Data Validation stage"
#
# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     data_validation = DataValidationTrainingPipeline()
#     data_validation.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
# except Exception as e:
#     logger.exception(e)
#     raise e
#
# STAGE_NAME = "Abstract Generalizer stage"
#
# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     generalizer = GeneralizerTrainingPipeline()
#     generalizer.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
# except Exception as e:
#     logger.exception(e)
#     raise e
#
# STAGE_NAME = "Solution stage"
#
# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     solution = SolutionTrainingPipeline()
#     solution.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
# except Exception as e:
#     logger.exception(e)
#     raise e
#
# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     internal_critique = InternalCritiqueTrainingPipeline()
#     internal_critique.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
# except Exception as e:
#     logger.exception(e)
#     raise e
#
# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     external_critique = ExternalCritiqueTrainingPipeline()
#     external_critique.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
# except Exception as e:
#     logger.exception(e)
#     raise e
#

from src.AINSTEIN import logger
from src.AINSTEIN.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.AINSTEIN.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.AINSTEIN.pipeline.stage_03_generalizer import GeneralizerTrainingPipeline
from src.AINSTEIN.pipeline.stage_04_solver import SolutionTrainingPipeline
from src.AINSTEIN.pipeline.stage_05_internal_critique import InternalCritiqueTrainingPipeline
from src.AINSTEIN.pipeline.stage_06_external_critique import ExternalCritiqueTrainingPipeline
from src.AINSTEIN.pipeline.stage_07_evaluation import EvaluationTrainingPipeline


class AINSTEINController:
    def __init__(
        self,
        max_internal_attempts: int = 3,
        max_external_attempts: int = 3,
        internal_status_path: str = "artifacts/internal_critique/internal_status.txt",
        external_status_path: str = "artifacts/external_critique/external_status.txt",
    ):
        self.max_internal_attempts = max_internal_attempts
        self.max_external_attempts = max_external_attempts
        self.internal_status_path = internal_status_path
        self.external_status_path = external_status_path

    def _run_stage(self, stage_name: str, stage_obj):
        try:
            logger.info(f">>>>>> stage {stage_name} started <<<<<<")
            result = stage_obj.main()
            logger.info(f">>>>>> stage {stage_name} completed <<<<<<")
            return result
        except Exception as e:
            logger.exception(e)
            raise e

    def _read_status(self, file_path: str) -> bool:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("Status:"):
                        value = line.split(":", 1)[1].strip()
                        return value.lower() == "true"
        except FileNotFoundError:
            logger.warning(f"Status file not found: {file_path}")
        return False

    def run(self):
        # Stage 1: Data ingestion
        self._run_stage("Data Ingestion stage", DataIngestionTrainingPipeline())

        # Stage 2: Data validation
        validation_passed = self._run_stage("Data Validation stage", DataValidationTrainingPipeline())
        if not validation_passed:
            raise ValueError("Data validation failed. Aborting pipeline.")

        # Stage 3: Generalizer runs ONCE
        logger.info("Generating fixed problem statement from abstract.")
        p_final = self._run_stage("Abstract Generalizer stage", GeneralizerTrainingPipeline())

        z_final = None

        # External critique loop
        for e in range(1, self.max_external_attempts + 1):
            logger.info(f"====== External Attempt {e}/{self.max_external_attempts} ======")

            # Internal critique loop
            for i in range(1, self.max_internal_attempts + 1):
                logger.info(f"------ Internal Attempt {i}/{self.max_internal_attempts} ------")

                # Solver regenerates solution each time
                z_candidate = self._run_stage("Solution stage", SolutionTrainingPipeline())

                # Internal critique checks solution
                self._run_stage("Internal Critique stage", InternalCritiqueTrainingPipeline())
                internal_pass = self._read_status(self.internal_status_path)

                if internal_pass:
                    logger.info("Internal critique passed.")
                    break

                logger.info("Internal critique failed. Sending solution back to Solver.")

            else:
                logger.info("Internal critique did not pass within max attempts. Moving to next external cycle.")
                continue

            # External critique checks solution
            self._run_stage("External Critique stage", ExternalCritiqueTrainingPipeline())
            external_pass = self._read_status(self.external_status_path)

            if external_pass:
                logger.info("External critique passed. Final solution accepted.")
                z_final = z_candidate
                break

            logger.info("External critique failed. Sending solution back to Solver.")

        if z_final is None:
            logger.warning("No final solution accepted after all critique attempts.")

        evaluation_result = self._run_stage("Evaluation stage", EvaluationTrainingPipeline())
        logger.info(f"Evaluation Result: {evaluation_result}")

        return p_final, z_final


if __name__ == "__main__":
    controller = AINSTEINController(
        max_internal_attempts=3,
        max_external_attempts=3,
        internal_status_path="artifacts/internal_critique/internal_status.txt",
        external_status_path="artifacts/external_critique/external_status.txt",
    )

    p_final, z_final = controller.run()

    logger.info(f"Final Problem Statement: {p_final}")
    logger.info(f"Final Solution: {z_final}")
