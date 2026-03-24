from typing import Optional, Literal

from itertools import product
from pydantic import BaseModel

from src.dataclasses.training_config import BaseTrainingConfig


class GridConfig(BaseModel):
    learning_rate: Optional[list[float]] = None
    batch_size: Optional[list[int]] = None
    optimizer: Optional[list[Literal["adam", "sgd", "rmsprop"]]] = None
    dropout: Optional[list[float]] = None
    weight_decay: Optional[list[float]] = None


class ExperimentConfig(BaseModel):
    name: str
    grid: GridConfig


    def generate_runs(self, base: BaseTrainingConfig):
        """
        Generates a series of training runs by creating variations of the input base configuration
        using a specified grid of parameter combinations.

        :param base: The base configuration for training. This configuration serves as the template
            that will be updated with different combinations of parameters based on the specified grid.
        :type base: BaseTrainingConfig

        :yield: A sequence of updated configurations for each combination of parameters in the grid.
        :rtype: Iterator[BaseTrainingConfig]
        """
        grid = {k: v for k, v in self.grid.model_dump().items() if v is not None}
        keys, values = zip(*grid.items())
        for combo in product(*values):
            override = dict(zip(keys, combo))
            yield base.model_copy(update=override)
