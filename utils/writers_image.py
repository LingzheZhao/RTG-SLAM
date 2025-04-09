"""
Log writer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Type

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from utils.base import InstantiateConfig

### save config
import shutil
import json
import os

# for exr
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


def to8b(x):
    """Converts a torch tensor to 8 bit"""
    return (255 * torch.clamp(x, min=0, max=1)).to(torch.uint8)


def to8b_np(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


@dataclass
class ImageWriterConfig(InstantiateConfig):
    """Configuration of image writer."""

    _target: Type = field(default_factory=lambda: ImageWriter)
    """The target class to be instantiated."""
    workdir: Path = Path("images")
    """Path to image save directory."""
    enabled: bool = True
    """Whether to enable image writer."""

    def __post_init__(self):
        self.workdir.mkdir(exist_ok=True)


class ImageWriter:
    """Image writer."""
    config: ImageWriterConfig

    def __init__(self, config: ImageWriterConfig):
        self.config = config

    def write_image(self, image_name: str, data: Tensor, rel_path: Optional[Path] = None):
        # check if image dimension is H, W, C
        data: Tensor = data.squeeze()
        if 2 == data.dim():
            data = data[None]
        try:
            assert 3 == data.dim()
        except:
            print(f"Wrong image shape {data.shape}")
            raise NotImplementedError
        if 3 == data.shape[0] or 1 == data.shape[0]:
            data = data.permute(1, 2, 0)
        try:
            assert 3 == data.shape[-1] or 1 == data.shape[-1]  # C
        except:
            print(f"Wrong image shape {data.shape}")
            raise NotImplementedError

        if self.config.enabled:
            image_dir = self.config.workdir
            if rel_path is not None:
                image_dir = image_dir / rel_path
            image_dir.mkdir(exist_ok=True, parents=True)
            if 'rgb' in image_name or 'blur' in image_name or "gt" in image_name or "mask" in image_name:
                path = str((image_dir / f"{image_name}").resolve())
                cv2.imwrite(path, cv2.cvtColor(to8b(data).cpu().numpy(), cv2.COLOR_RGB2BGR))
            else:
                path = str((self.config.workdir / f"{image_name}").resolve())
                cv2.imwrite(path, data.cpu().numpy())
            print(f"Saved image to {path}")


def save_config(cfg):
    print("Saving config and script...")
    save_path = Path(cfg["data"]["output"])
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    # shutil.copy(__file__, save_path / Path(__file__).name)

    with open(os.path.join(save_path, 'config.json'), "w", encoding='utf-8') as f:
        f.write(json.dumps(cfg, indent=4))