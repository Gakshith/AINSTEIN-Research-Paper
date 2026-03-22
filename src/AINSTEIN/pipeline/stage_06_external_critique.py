from src.AINSTEIN.config.configuration import ConfigurationManager
from src.AINSTEIN.components.external_critique import ExternalCritique
from src.AINSTEIN import logger

STAGE_NAME = "External Critique stage"

class ExternalCritiqueTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config_manager = ConfigurationManager()
        external_critique_config = config_manager.get_external_critique_config()

        status = "False"
        with open(external_critique_config.internal_critique_status, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Status:"):
                    status = line.split(":", 1)[1].strip()
                    break

        if status != "True":
            logger.info("Skipping external critique because internal critique did not pass.")
            return None

        external_critique = ExternalCritique(external_critique_config)
        return external_critique.get_external_critique_score()
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ExternalCritiqueTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e



