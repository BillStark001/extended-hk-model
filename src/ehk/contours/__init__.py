"""Recoverable sequential estimation of contours and decision boundaries."""

from .acquisition import AcquisitionResult, Action, propose_action
from .adapters import (
    LandscapeBarrierGapEvaluator,
    OperatorGapEvaluator,
    PairedMultimetricEvaluator,
    common_terminal_time,
    hellinger_gap,
    normalized_time_l1_gap,
    terminal_density_l2_gaps,
    total_variation_from_counts,
)
from .gp import MaternGP
from .protocol import ContourProtocol, load_protocol
from .runner import SequentialContourEstimator
from .store import Evaluation, EvaluationStore

__all__ = [
    "AcquisitionResult",
    "Action",
    "ContourProtocol",
    "Evaluation",
    "EvaluationStore",
    "LandscapeBarrierGapEvaluator",
    "MaternGP",
    "OperatorGapEvaluator",
    "PairedMultimetricEvaluator",
    "SequentialContourEstimator",
    "common_terminal_time",
    "hellinger_gap",
    "load_protocol",
    "normalized_time_l1_gap",
    "propose_action",
    "terminal_density_l2_gaps",
    "total_variation_from_counts",
]
