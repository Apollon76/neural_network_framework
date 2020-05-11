from dataclasses import dataclass
from typing import Dict, Optional

from bokeh.models import ColumnDataSource
from bokeh.plotting import Figure
from pydantic import BaseModel


class TimeMetric(BaseModel):
    last_duration: Optional[float]
    average_duration: float
    epoch: Optional[int]


class EpochMetric(BaseModel):
    epoch: int
    value: float


class Metrics(BaseModel):
    fit_metrics: TimeMetric
    fit_batch_metrics: TimeMetric
    gradients_metrics: TimeMetric
    train_score: EpochMetric
    train_loss: EpochMetric
    validation_score: Optional[EpochMetric]
    validation_loss: Optional[EpochMetric]
    apply_gradients_metrics: Dict[str, TimeMetric]
    backward_pass_metrics: Dict[str, TimeMetric]
    forward_pass_metrics: Dict[str, TimeMetric]
    gradient_step_metrics: Dict[str, TimeMetric]


@dataclass
class EpochPlot:
    plot: Figure
    source: ColumnDataSource


@dataclass
class TrainTestPlots:
    score: EpochPlot
    loss: EpochPlot


@dataclass
class ProfilingPlots:
    full_epoch: EpochPlot


def load_metrics(filename: str) -> Metrics:
    with open(filename) as f:
        return Metrics.parse_raw(f.read())
