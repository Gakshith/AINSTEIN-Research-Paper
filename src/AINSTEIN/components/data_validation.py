import pandas as pd
import numpy as np
from src.AINSTEIN import logger
from src.AINSTEIN.entity.config_entity import DataValidationConfig
import os
class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_dataset(self):
        dataset = pd.read_csv(self.config.data_path)
        validation_status = True
        os.makedirs(os.path.dirname(self.config.STATUS_FILE), exist_ok=True)
        null_value_counts = dataset.isnull().sum().values
        if np.count_nonzero(null_value_counts) > 0:
            validation_status = False
            logger.warning("There are null values in the dataset.")
            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Status: {validation_status}")
            return validation_status
        for column in self.config.all_schema.keys():
            if column not in dataset.columns:
                validation_status = False
                with open(self.config.STATUS_FILE, "w") as f:
                    f.write(f"Status: {validation_status}")
                break

        with open(self.config.STATUS_FILE, "w") as f:
            f.write(f"Status: {validation_status}")

        logger.info(f"Validation status: {validation_status}")
        return validation_status
