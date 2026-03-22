from src.AINSTEIN.config.configuration import ConfigurationManager
from src.AINSTEIN.components.generalizer import Generalizer
from src.AINSTEIN import logger

STAGE_NAME = "Abstract Generalizer stage"

class GeneralizerTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config_manager = ConfigurationManager()
        generalizer_config = config_manager.get_generalizer_config()
        generalizer = Generalizer(generalizer_config)
        abstract_text = generalizer.load_abstract()
        generalized_problem = generalizer.generalization_agent(abstract_text)
        return generalized_problem.generalized_research_abstract
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = GeneralizerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e







