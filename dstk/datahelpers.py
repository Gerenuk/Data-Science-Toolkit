from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class Bins:
    bin_min: float
    bin_max: float
    bin_num: int
    width: float = None
    edges: Any = None
    torch_kwargs: dict = None

    def __post_init__(self):
        self.width = (self.bin_max - self.bin_min) / self.bin_num
        self.edges = np.linspace(self.bin_min, self.bin_max, self.bin_num)
        self.torch_kwargs = {
            "bins": self.bin_num,
            "min": self.bin_min,
            "max": self.bin_max,
        }

    def __getitem__(self, idx):
        return self.bin_min + idx * self.width
