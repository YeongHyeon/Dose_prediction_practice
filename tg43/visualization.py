"""Visualisation helpers for TG-43 dose and CT overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

import tg43.utils as utils

def plot_dwell_times_comparison(
    original_dwells: Sequence[Any],
    optimized_dwells: Sequence[Any],
    reference_label: str = "Baseline",
    candidate_label: str = "IPSA",
    save_dir: Optional[Union[str, Path]] = None,
    save_name: Optional[str] = None,
    extension: str = "png",
) -> None:
    """Plot stacked dwell times for baseline vs. candidate plans (optionally saving to disk)."""

    summed_original = np.array([dp.dwell_time_s for dp in original_dwells], dtype=float)
    summed_optimized = np.array([dp.dwell_time_s for dp in optimized_dwells], dtype=float)

    if summed_original.size != summed_optimized.size:
        raise ValueError("Dwell sequences must have matching lengths.")
    if summed_original.size % 2 != 0:
        raise ValueError("Dwell sequences must contain an even number of entries.")

    combined_original = summed_original[0::2] + summed_original[1::2]
    combined_optimized = summed_optimized[0::2] + summed_optimized[1::2]
    indices = range(len(combined_original))

    plt.figure(figsize=(8, 4), dpi=150)
    plt.bar(indices, combined_original, label=reference_label, alpha=0.6)
    plt.bar(indices, combined_optimized, label=candidate_label, alpha=0.6)
    plt.xlabel("Dwell position index")
    plt.ylabel("Dwell time [s]")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left")
    plt.tight_layout()

    if save_dir is not None:
        if save_name is None:
            raise ValueError("save_name must be provided when save_dir is specified.")
        plt.savefig(save_dir / f"{save_name}.{extension}")
        plt.close()
    else:
        plt.show()

    plt.figure(figsize=(8, 4), dpi=150)
    plt.subplot(1, 2, 1)
    plt.title(reference_label, loc='left')
    plt.bar(indices, combined_original, label=reference_label, alpha=0.6)
    plt.xlabel("Dwell position index")
    plt.ylabel("Dwell time [s]")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.title(candidate_label, loc='left')
    plt.bar(indices, combined_optimized, label=candidate_label, alpha=0.6)
    plt.xlabel("Dwell position index")
    plt.ylabel("Dwell time [s]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir is not None:
        if save_name is None:
            raise ValueError("save_name must be provided when save_dir is specified.")
        plt.savefig(save_dir / f"{save_name}_separated.{extension}")
        plt.close()
    else:
        plt.show()




def plot_optimization_history(
    history: Iterable[Dict[str, float]],
    *,
    save_dir: Optional[Union[str, Path]] = None,
    save_name: Optional[str] = None,
    extension: str = "png",
) -> None:
    """Plot optimisation objective traces (and temperature, if recorded) and optionally save them."""

    history = list(history)
    if not history:
        return
    if save_dir is not None and save_name is None:
        raise ValueError("save_name must be provided when save_dir is specified.")

    steps = [entry["step"] for entry in history]
    objective = [entry["objective"] for entry in history]

    plt.figure(figsize=(8, 4), dpi=150)
    plt.plot(steps, objective, linewidth=1)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(save_dir / f"{save_name}_cost.{extension}")
        plt.close()
    else:
        plt.show()


def plot_dvh_curves(
    dvh_reference: Dict[str, Any],
    dvh_candidate: Dict[str, Any],
    *,
    reference_label: str = "Baseline",
    candidate_label: str = "IPSA",
    save_dir: Optional[Union[str, Path]] = None,
    save_name: Optional[str] = None,
    extension: str = "png",
) -> None:
    """Plot baseline vs. candidate DVH curves per structure (dual layout + zoomed view)."""

    if not dvh_reference or not dvh_candidate:
        return

    save_dir = Path(save_dir) if save_dir is not None else None
    structures = list(dvh_reference.keys())
    colours = list(mcolors.TABLEAU_COLORS.values())

    def _plot(suffix: str, xlim: Optional[Sequence[float]] = None, ylim: Optional[Sequence[float]] = None) -> None:
        width = 12 if suffix == "full" else 4
        plt.figure(figsize=(width, 4), dpi=150)
        for idx, name in enumerate(structures):
            if name not in dvh_candidate:
                continue
            colour = colours[idx % len(colours)]
            ref = dvh_reference[name]
            cand = dvh_candidate[name]
            plt.plot(
                ref.dose_bins_Gy,
                ref.volume_percent,
                color=colour,
                linewidth=1.5,
                linestyle="-",
                label=f"{name} ({reference_label})",
            )
            plt.plot(
                cand.dose_bins_Gy,
                cand.volume_percent,
                color=colour,
                linewidth=1.5,
                linestyle=":",
                label=f"{name} ({candidate_label})",
            )
        plt.xlabel("Dose [Gy]")
        plt.ylabel("Volume [%]")
        plt.grid(True)
        if suffix == "full":
            plt.legend(bbox_to_anchor=(0.5, -0.15), ncol=5, loc="upper center")
        if xlim:
            plt.xlim(*xlim)
        if ylim:
            plt.ylim(*ylim)
        plt.tight_layout()
        if save_dir is not None:
            if save_name is None:
                raise ValueError("save_name must be provided when save_dir is specified.")
            plt.savefig(save_dir / f"{save_name}_{suffix}.{extension}")
            plt.close()
        else:
            plt.show()

    _plot("full", xlim=(-0.05, 40))
    _plot("zoom_high", xlim=(5, 8), ylim=(80, 101))
    _plot("zoom_low", xlim=(-0.05, 3))


def plot_dose_distribution(
    ct_array: np.ndarray,
    dose_array: np.ndarray,
    ct_metadata: Dict[str, Any],
    dose_max: Optional[float] = None,
    save_dir: Optional[Union[str, Path]] = None,
    save_name: Optional[str] = None,
    extension: str = "png",
) -> None:
    """Overlay summed dose projections on CT views and optionally save them.

    Parameters
    ----------
    ct_array : numpy.ndarray
        CT voxel intensities arranged as ``(z, y, x)``.
    dose_array : numpy.ndarray
        Dose values aligned with the CT grid.
    ct_metadata : dict[str, Any]
        Dictionary containing ``size`` and ``spacing`` entries describing the CT geometry.
    dose_max : float, optional
        Upper clip limit for the dose overlay. When ``None`` the raw dose grid is used.
    save_dir : str or Path, optional
        Directory where the figure is written. When omitted the plot is shown interactively.
    save_name : str, optional
        Basename used when ``save_dir`` is provided. Required if ``save_dir`` is given.
    extension : str, optional
        File extension (without dot) used when saving the figure.

    Raises
    ------
    ValueError
        If ``save_dir`` is provided without ``save_name``.
    """

    dose_overlay = utils.dose_clip(dose_array, 0, dose_max) if dose_max is not None else dose_array
    # np.save(save_dir / f"{save_name}.npy", dose_overlay)

    x_ticks = range(ct_metadata["size"][0])
    y_ticks = range(ct_metadata["size"][1])
    z_ticks = range(ct_metadata["size"][2])

    x_coords_mm = np.array(x_ticks) * ct_metadata["spacing"][0]
    y_coords_mm = np.array(y_ticks) * ct_metadata["spacing"][1]
    z_coords_mm = np.array(z_ticks) * ct_metadata["spacing"][2]

    tick_step = 100
    z_tick_step = max(1, tick_step // 2)

    plt.figure(figsize=(12, 4.5), dpi=100)

    plt.subplot(1, 3, 1)
    plt.title("Axial view", loc="left")
    plt.imshow(np.sum(ct_array[::-1], axis=0), aspect="auto", cmap="gray")
    plt.imshow(np.sum(dose_overlay[::-1], axis=0), aspect="auto", cmap="jet", alpha=0.5)
    plt.xticks(ticks=x_ticks[::tick_step], labels=np.round(x_coords_mm[::tick_step], 1), rotation=90)
    plt.yticks(ticks=y_ticks[::tick_step], labels=np.round(y_coords_mm[::tick_step], 1))
    plt.xlabel("Right-Left (mm)")
    plt.ylabel("Anterior-Posterior (mm)")
    plt.grid(linewidth=0.3, color="lime", linestyle="--")

    plt.subplot(1, 3, 2)
    plt.title("Coronal view", loc="left")
    plt.imshow(np.sum(ct_array[::-1], axis=1), aspect="auto", cmap="gray")
    plt.imshow(np.sum(dose_overlay[::-1], axis=1), aspect="auto", cmap="jet", alpha=0.5)
    plt.xticks(ticks=x_ticks[::tick_step], labels=np.round(x_coords_mm[::tick_step], 1), rotation=90)
    plt.yticks(ticks=z_ticks[::z_tick_step], labels=np.round(z_coords_mm[::z_tick_step], 1))
    plt.xlabel("Right-Left (mm)")
    plt.ylabel("Superior-Inferior (mm)")
    plt.grid(linewidth=0.3, color="lime", linestyle="--")

    plt.subplot(1, 3, 3)
    plt.title("Sagittal view", loc="left")
    plt.imshow(np.sum(ct_array[::-1], axis=2), aspect="auto", cmap="gray")
    plt.imshow(np.sum(dose_overlay[::-1], axis=2), aspect="auto", cmap="jet", alpha=0.5)
    plt.xticks(ticks=y_ticks[::tick_step], labels=np.round(y_coords_mm[::tick_step], 1), rotation=90)
    plt.yticks(ticks=z_ticks[::z_tick_step], labels=np.round(z_coords_mm[::z_tick_step], 1))
    plt.xlabel("Anterior-Posterior (mm)")
    plt.ylabel("Superior-Inferior (mm)")
    plt.grid(linewidth=0.3, color="lime", linestyle="--")

    plt.tight_layout()
    if save_dir is not None:
        if save_name is None:
            raise ValueError("save_name must be provided when save_dir is specified.")
        plt.savefig(save_dir / f"{save_name}.{extension}")
        plt.close()
    else:
        plt.show()

def visualize_coin_values(
    list_coin: List[Dict[str, Any]],
    save_dir: Optional[Union[str, Path]] = None,
    save_name: Optional[str] = None,
) -> None:
    
    plt.figure(figsize=(6, 4), dpi=150)
    plt.scatter(range(len(list_coin)), list_coin)
    plt.xlabel("Solution index")
    plt.ylabel("COIN")
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_dir / f"{save_name}_coin_values.png")
    plt.close()

def visualize_pareto_front(
    solutions: List[Dict[str, Any]],
    save_dir: Optional[Union[str, Path]] = None,
    save_name: Optional[str] = None,
) -> None:
    
    if(type(solutions[0]) is list):
        solutions_ = []
        for _, pair in enumerate(solutions):
            solutions_.extend(pair)
        solutions = solutions_

    dict_penalty = {}
    for name_spec in list(solutions[0]["best_state"]["breakdown"].keys()):
        dict_penalty[name_spec] = []

    list_target = []
    list_oar = []
    for _, solution in enumerate(solutions):
        for name_spec in list(solution["best_state"]["breakdown"].keys()):
            role = solution["best_state"]["breakdown"][name_spec]["role"]
            raw = solution["best_state"]["breakdown"][name_spec]["raw"]
            dict_penalty[name_spec].append(raw)

            if role == "target" and name_spec not in list_target:
                list_target.append(name_spec)
            elif role == "oar" and name_spec not in list_oar:
                list_oar.append(name_spec)


    list_keys = list(dict_penalty.keys())
    for idx_1, name_spec_1 in enumerate(list_keys):
        for idx_2, name_spec_2 in enumerate(list_keys):
            if idx_1 >= idx_2: continue
            if name_spec_1 == name_spec_2: continue

            plt.figure(figsize=(6, 6), dpi=150)
            plt.scatter(
                dict_penalty[name_spec_1],
                dict_penalty[name_spec_2],
                alpha=0.7,
                s=20,
                c='#1972ff',
                edgecolors='#1972ff'
            )
            plt.xlabel(f"{name_spec_1} penalty")
            plt.ylabel(f"{name_spec_2} penalty")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_dir / f"{save_name}_{name_spec_1}_{name_spec_2}.png")
            plt.close()

    tmp_target = None
    tmp_oar = None
    for name_spec in list(dict_penalty.keys()):
        if name_spec in list_target:
            if tmp_target is None:
                tmp_target = np.array(dict_penalty[name_spec].copy())
            else:
                tmp_target += np.array(dict_penalty[name_spec].copy())
        if name_spec in list_oar:
            if tmp_oar is None:
                tmp_oar = np.array(dict_penalty[name_spec].copy())
            else:
                tmp_oar += np.array(dict_penalty[name_spec].copy())
    plt.figure(figsize=(6, 6), dpi=150)
    plt.scatter(
        tmp_target,
        tmp_oar,
        alpha=0.7,
        s=20,
        c='#1972ff',
        edgecolors='#1972ff'
    )
    plt.xlabel(f"Total target penalty")
    plt.ylabel(f"Total OAR penalty")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"{save_name}_target_total_oar.png")
    plt.close()
