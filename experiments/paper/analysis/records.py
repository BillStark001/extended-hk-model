from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy import Integer, Float, String, Text
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from typing import Optional

from ehk.common.storage.sqlite import NumpyArrayType

if TYPE_CHECKING:
    from ehk.micro.scenarios import ScenarioMetadata

Base = declarative_base()


def _():
    pass


exclude = [
    "name",
    "origin",
    "tolerance",
    "decay",
    "rewiring",
    "retweet",
    "recsys_type",
    "tweet_retain_count",
]


class ScenarioStatistics(Base):
    __tablename__ = "scenariostatistics"

    Base = Base

    # metadata fields
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    origin: Mapped[str] = mapped_column(String(255), index=True, nullable=False)

    tolerance: Mapped[float] = mapped_column(Float, index=True, nullable=False)
    decay: Mapped[float] = mapped_column(Float, index=True, nullable=False)
    rewiring: Mapped[float] = mapped_column(Float, index=True, nullable=False)
    retweet: Mapped[float] = mapped_column(Float, index=True, nullable=False)
    recsys_type: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    tweet_retain_count: Mapped[float] = mapped_column(Float, index=True, nullable=False)

    # simulation result fields
    step: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    active_step: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    active_step_threshold: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    g_index_mean_active: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    x_indices: Mapped[Optional[np.ndarray]] = mapped_column(
        NumpyArrayType(dtype=np.int32), nullable=True
    )
    h_index: Mapped[Optional[np.ndarray]] = mapped_column(
        NumpyArrayType(dtype=np.float64), nullable=True
    )
    p_index: Mapped[Optional[np.ndarray]] = mapped_column(
        NumpyArrayType(dtype=np.float64), nullable=True
    )
    g_index: Mapped[Optional[np.ndarray]] = mapped_column(
        NumpyArrayType(dtype=np.float64), nullable=True
    )

    grad_index: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    event_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    event_step_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    triads: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    x_bc_hom: Mapped[Optional[np.ndarray]] = mapped_column(
        NumpyArrayType(dtype=np.int32), nullable=True
    )
    bc_hom_smpl: Mapped[Optional[np.ndarray]] = mapped_column(
        NumpyArrayType(dtype=np.float64), nullable=True
    )

    x_mean_vars: Mapped[Optional[np.ndarray]] = mapped_column(
        NumpyArrayType(dtype=np.int32), nullable=True
    )
    mean_vars_smpl: Mapped[Optional[np.ndarray]] = mapped_column(
        NumpyArrayType(dtype=np.float64), nullable=True
    )

    opinion_diff: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    x_opinion_diff_mean: Mapped[Optional[np.ndarray]] = mapped_column(
        NumpyArrayType(dtype=np.int32), nullable=True
    )
    opinion_diff_mean_smpl: Mapped[Optional[np.ndarray]] = mapped_column(
        NumpyArrayType(dtype=np.float64), nullable=True
    )

    last_community_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    last_community_sizes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    last_opinion_peak_count: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )

    p_backdrop: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    h_backdrop: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    g_backdrop: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    opinion_diff_seg_mean: Mapped[Optional[np.ndarray]] = mapped_column(
        NumpyArrayType(dtype=np.float64), nullable=True
    )
    opinion_diff_seg_std: Mapped[Optional[np.ndarray]] = mapped_column(
        NumpyArrayType(dtype=np.float64), nullable=True
    )


def stats_from_dict(
    scenario_metadata: "ScenarioMetadata",
    stats_dict: dict,
    origin: str,
) -> ScenarioStatistics:
    _d = {}
    for k, v in stats_dict.items():
        if k in exclude:
            continue
        if hasattr(ScenarioStatistics, k):
            _d[k] = v

    scenario_name = scenario_metadata["UniqueName"]
    instance = ScenarioStatistics(
        name=scenario_name,
        origin=origin,
        tolerance=scenario_metadata["Tolerance"],
        decay=scenario_metadata["Influence"],
        rewiring=scenario_metadata["RewiringRate"],
        retweet=scenario_metadata["RepostRate"],
        recsys_type=scenario_metadata["RecsysFactoryType"],
        tweet_retain_count=scenario_metadata["PostRetainCount"],
        **_d,
    )
    return instance
