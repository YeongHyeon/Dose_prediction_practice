"""Generic numerical utilities used across TG-43 examples."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

EPSILON = 1e-6

def min_max_normalization(array: np.ndarray) -> np.ndarray:
    """Normalize values into ``[0, 1]`` while guarding against zero range.

    Parameters
    ----------
    array : numpy.ndarray
        Input values to be normalised.

    Returns
    -------
    numpy.ndarray
        Array with entries scaled into ``[0, 1]``. Constant inputs map to ones
        thanks to a small epsilon in the numerator and denominator.
    """

    array = np.asarray(array, dtype=float)

    return (array - array.min() + EPSILON) / (array.max() - array.min() + EPSILON)

def measure_l2_error(target: np.ndarray, compare: np.ndarray) -> np.ndarray:
    """Return element-wise L2 error between ``target`` and ``compare`` arrays.

    Parameters
    ----------
    target : numpy.ndarray
        Reference values used to measure the deviation.
    compare : numpy.ndarray
        Values compared against ``target``.

    Returns
    -------
    numpy.ndarray
        Element-wise absolute difference matching the broadcasted shape of the inputs.
    """

    target = np.asarray(target, dtype=float)
    compare = np.asarray(compare, dtype=float)
    difference = target - compare
    error = np.sqrt(difference ** 2)

    return error


def save_dvh(dvh_map: Dict, path: Path | str) -> None:
    """Persist DVH curves to a compressed ``.npz`` archive.

    Parameters
    ----------
    dvh_map : dict
        Mapping of structure name to DVH objects containing arrays to serialise.
    path : str or Path
        Basename used for the emitted ``.npz`` and ``.csv`` files (without extension).
    """

    dic_dvh = {}
    for name, dvh in dvh_map.items():
        prefix = name.replace(" ", "_")
        dic_dvh[f"{prefix}_dose_Gy"] = dvh.dose_bins_Gy
        dic_dvh[f"{prefix}_volume_percent"] = dvh.volume_percent
        if dvh.volume_cc is not None:
            dic_dvh[f"{prefix}_volume_cc"] = dvh.volume_cc

    np.savez_compressed(f"{path}.npz", **dic_dvh)
    pd.DataFrame.from_dict(dic_dvh).to_csv(f"{path}.csv", index=False)
