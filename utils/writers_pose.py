"""
Log writer.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Type

import numpy as np
import pypose as pp
import torch
from jaxtyping import Float
from pypose import LieTensor
from torch import Tensor

from utils.base import InstantiateConfig

### save config
import shutil
import json
import os


@dataclass
class TrajectoryWriterConfig(InstantiateConfig):
    """Config for abstract class for traj writer"""
    _target: Type = field(default_factory=lambda: TrajectoryWriter)
    """The target class to be instantiated."""
    filename: Path = Path("trajectory.txt")
    """Path to tum trajectory file."""

    def __post_init__(self):
        self.filename.parent.mkdir(exist_ok=True)

class TrajectoryWriter:
    """Abstract class for traj writer"""
    config: TrajectoryWriterConfig
    data: Dict[int | float, Float[Tensor, "4 4"]]

    def __init__(self, config: TrajectoryWriterConfig):
        self.config = config
        """Config"""
        self.data = {}
        """Data dict format: {timestamp: data}"""
    
    @abstractmethod
    def write(
        self,
        timestamp: int | float,
        pose: Float[LieTensor, "1 7"] | Float[Tensor, "4 4"]
    ):
        """Write pose to trajectory file"""
        assert isinstance(pose, LieTensor) or \
               isinstance(pose, Tensor) and \
                   pose.shape[-1] == 4


@dataclass
class NumpyTrajectoryWriterConfig(TrajectoryWriterConfig):
    """Config for numpy traj writer"""
    _target: Type = field(default_factory=lambda: NumpyTrajectoryWriter)
    """The target class to be instantiated."""
    filename: Path = Path("trajectory.npz")
    """Path to NPZ trajectory file."""

class NumpyTrajectoryWriter(TrajectoryWriter):
    """Numpy trajectory writer"""
    config: NumpyTrajectoryWriterConfig
    data: Dict[int | float, Float[LieTensor, "1 7"]]

    def __init__(self, config: NumpyTrajectoryWriterConfig):
        super().__init__(config)
    
    def write(
            self,
            timestamp: int | float,
            pose: Float[LieTensor, "1 7"] | Float[Tensor, "4 4"]
    ):
        """Write pose to numpy trajectory file."""
        if isinstance(pose, LieTensor):
            pose = pose.matrix().tensor()
        if pose.shape[-1] == 4:
            if pose.shape[-2] == 3:
                row_shape = pose.shape
                row_shape[-2] = 1
                row = torch.zeros(row_shape)
                row[..., -1, -1] = 1
                pose = torch.cat(pose, row)
                assert pose.shape[-2] == 4
        pose = pose.squeeze()
        assert pose.dim() == 2
        self.data[timestamp] = pose
        all_data = torch.stack(list(self.data.values()))
        np.savez_compressed(self.config.filename, all_data)


@dataclass
class TumTrajectoryWriterConfig(TrajectoryWriterConfig):
    """Configuration of tum trajectory."""

    _target: Type = field(default_factory=lambda: TumTrajectoryWriter)
    """The target class to be instantiated."""
    filename: Path = Path("trajectory_tum.txt")
    """Path to tum trajectory file."""
    first_line: str = "# timestamp tx ty tz qx qy qz qw"
    """First line of tum trajectory file."""


class TumTrajectoryWriter(TrajectoryWriter):
    """Tum trajectory writer."""
    config: TumTrajectoryWriterConfig
    data: Dict[int | float, Float[LieTensor, "1 7"]]

    def __init__(self, config: TumTrajectoryWriterConfig):
        super().__init__(config)
        with open(self.config.filename, 'w') as f:
            f.write(f"{self.config.first_line}\n")

    def write(
            self,
            timestamp: int | float,
            pose: Float[LieTensor, "1 7"] | Float[Tensor, "4 4"]
    ):
        """Write pose to tum trajectory file."""
        if isinstance(pose, LieTensor):
            pose = pose.tensor()
        if pose.shape[-1] == 4:
            pose = pp.mat2SE3(pose).tensor().squeeze()
        pose = pose.squeeze()
        assert pose.dim() == 1
        self.data[timestamp] = pose
        with open(self.config.filename, 'a+', buffering=1) as f:
            pose_str = " ".join([str(x) for x in pose.data.cpu().numpy()])
            f.write(f"{timestamp} {pose_str}\n")
