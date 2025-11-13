"""Generic numerical utilities used across TG-43 examples."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

EPSILON = 1e-6

def min_max_normalization(array: np.ndarray) -> np.ndarray:
    """Scale an array into ``[0, 1]`` with epsilon guards for constant inputs."""

    array = np.asarray(array, dtype=float)

    return (array - array.min() + EPSILON) / (array.max() - array.min() + EPSILON)

def measure_l2_error(target: np.ndarray, compare: np.ndarray) -> np.ndarray:
    """Return the element-wise L2 error between ``target`` and ``compare``."""

    target = np.asarray(target, dtype=float)
    compare = np.asarray(compare, dtype=float)
    difference = target - compare
    error = np.sqrt(difference ** 2)

    return error


def save_dvh(dvh_map: Dict, path: Path | str) -> None:
    """Persist DVH curves to both ``.npz`` (binary) and ``.csv`` snapshots."""

    dic_dvh = {}
    for name, dvh in dvh_map.items():
        prefix = name.replace(" ", "_")
        dic_dvh[f"{prefix}_dose_Gy"] = dvh.dose_bins_Gy
        dic_dvh[f"{prefix}_volume_percent"] = dvh.volume_percent
        if dvh.volume_cc is not None:
            dic_dvh[f"{prefix}_volume_cc"] = dvh.volume_cc

    np.savez_compressed(f"{path}.npz", **dic_dvh)
    pd.DataFrame.from_dict(dic_dvh).to_csv(f"{path}.csv", index=False)
