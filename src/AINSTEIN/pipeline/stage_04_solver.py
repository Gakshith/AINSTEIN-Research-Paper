from src.AINSTEIN.config.configuration import ConfigurationManager
from src.AINSTEIN.components.solver import Solver
from src.AINSTEIN import logger

STAGE_NAME = "Solution stage"

class SolutionTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config_manager = ConfigurationManager()
        solver_config = config_manager.get_solver_config()
        solver = Solver(solver_config)
        solution = solver.get_solution()
        return solution.research_solution
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = SolutionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e







