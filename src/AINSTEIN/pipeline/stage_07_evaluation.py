from src.AINSTEIN.config.configuration import ConfigurationManager
from src.AINSTEIN.components.evaluation import Evaluation
from src.AINSTEIN import logger

STAGE_NAME = "Evaluation stage"


class EvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        evaluation_config = config_manager.get_evaluation_config()
        evaluation = Evaluation(evaluation_config)
        return evaluation.evaluate_solution()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
