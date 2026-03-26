import math
from typing import Callable

import numpy as np
import optuna
from optuna.samplers import GridSampler, TPESampler

from src.dataclasses.experiment_config import ExperimentConfig


class HyperparameterTuner:

    def __init__(self, experiment_config: ExperimentConfig, objective_fn: Callable):
        self.config = experiment_config
        self.objective_fn = objective_fn
        self.study = None

    def _build_sampler(self):
        if self.config.search_strategy == "grid":
            search_space = {
                p.name: np.linspace(p.min, p.max, p.number_of_entries).tolist()
                for p in self.config.tune_parameter
            }
            return GridSampler(search_space), math.prod(
                p.number_of_entries for p in self.config.tune_parameter
            )
        else:
            return TPESampler(), self.config.n_trials

    def tune(self):
        sampler, n_trials = self._build_sampler()
        self.study = optuna.create_study(sampler=sampler, direction=self.config.direction)
        self.study.optimize(self.objective_fn, n_trials=n_trials)
        return self.study.best_params
