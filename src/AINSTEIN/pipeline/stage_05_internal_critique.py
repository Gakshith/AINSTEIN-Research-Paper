from src.AINSTEIN.config.configuration import ConfigurationManager
from src.AINSTEIN.components.internal_critique import InternalCritique
from src.AINSTEIN import logger

STAGE_NAME = "Internal Critique stage"

class InternalCritiqueTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config_manager = ConfigurationManager()
        internal_critique_config = config_manager.get_internal_critique_config()
        internal_critique = InternalCritique(internal_critique_config)
        return internal_critique.get_internal_critique_score()
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = InternalCritiqueTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e



