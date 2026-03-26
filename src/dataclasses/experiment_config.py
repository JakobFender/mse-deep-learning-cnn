from typing import Literal, Optional

from pydantic import BaseModel, model_validator


class TuneParameter(BaseModel):
    name: str
    min: float
    max: float
    number_of_entries: Optional[int] = None

class ExperimentConfig(BaseModel):
    search_strategy: Literal["grid", "bayesian"] = "bayesian"
    tune_parameter: list[TuneParameter]
    n_trials: int
    direction: str = "maximize"

    @model_validator(mode="after")
    def check_number_of_entries(self):
        if self.search_strategy == "grid":
            missing = [p.name for p in self.tune_parameter if p.number_of_entries is None]
            if missing:
                raise ValueError(f"Parameters {missing} must have number_of_entries set when strategy is 'grid'.")
        return self
