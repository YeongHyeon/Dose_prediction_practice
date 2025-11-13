import copy
import numpy as np

from skimage.feature import peak_local_max

import tg43.dicom_helper as dhelp
import tg43.contour_helper as chelp
import tg43.dose_calculation as dosecal
from typing import Any, Dict, Iterable, Tuple, List

def load_case(
    paths: Dict[str, str], 
    cfg: Dict[str, Any] = None,
):

    ## CT
    ct_image, ct_array, _ = dhelp.load_ct_volume(paths["ct"])

    ## RTPLAN
    rt_channels = dhelp.load_rtplan_by_channel(paths["plan"])

    ## RTSTRUCT
    mask_array, mask_names = load_mask_array(paths["struct"], ct_image, ct_array)

    ## RTDOSE
    dose_array = load_dose_array(paths['dose'], ct_image, rt_channels, cfg)


    ## Slice extraction
    ct_slices, dose_slices, mask_slices, dwell_positions, dwell_candidates = extract_slices_by_dwell_positions(
        ct_image=ct_image,
        ct_array=ct_array,
        dose_array=dose_array,
        mask_array=mask_array,
        rt_channels=rt_channels,
    )

    return {
        "ct_slices":ct_slices, 
        "dose_slices":dose_slices, 
        "mask_slices":mask_slices, 
        "mask_names":mask_names, 
        "dwell_positions":dwell_positions,
        "dwell_candidates":dwell_candidates
    }

def load_dose_array(
    paths: Dict[str, str],
    ct_image,
    rt_channels,
    cfg: Dict[str, Any],
):
    mode = cfg["hyperparams"]["dwell_time_mode"]
    scale = cfg["hyperparams"]["scale"]

    if mode == "perturb":
        pass
    elif mode == "random":
        pass
    else:
        dose_image, _, _ = dhelp.load_rtdose_volume(paths['dose'])
        _, dose_array, _ = dhelp.resample_to_reference(dose_image, ct_image)
        return dose_array

    ## Generate RTDOSE 
    rt_channels = adjust_channel_dwells(
        rt_channels=rt_channels,
        mode=mode,
        noise_scale=scale,
    )
    dose_array = dosecal.calculate_and_resample_to_ct(
        ct_image=ct_image,
        channels=rt_channels,
        anisotropy_path=cfg["hyperparams"]["anisotropy_table"],
        radial_path=cfg["hyperparams"]["radial_table"],
        grid_spacing_mm=ct_image.GetSpacing(),
        margin_mm=20.0,
        max_distance_cm=10.0,
    )

    return dose_array.resampled_array


def load_mask_array(
    path, 
    ct_image, 
    ct_array
):
    
    ## RTSTRUCT
    rtstruct = chelp.load_rtstruct(path)

    ## Rasterize the masks 
    structures = ["Bladder", "Rectum", "Sigmoid", "Bowel", "HR-CTV"]
    num_axial, num_coronal, num_sagittal = ct_array.shape
    num_class = len(structures)
    mask_array = np.zeros((num_axial, num_coronal, num_sagittal, num_class))
    for idx_structure, name_structure in enumerate(structures):
        mask_tmp = chelp.rasterise_structure(rtstruct, name_structure, ct_image)
        mask_array[..., idx_structure] = mask_tmp.astype(np.float32)

    return mask_array, structures

def extract_slices_by_dwell_positions(
    ct_image,
    ct_array,
    dose_array,
    mask_array,
    rt_channels
):

    ## Extract slices from volume by dwell positions
    ct_slices = []
    dose_slices = []
    mask_slices = []
    dwell_positions = dhelp.extract_dwell_positions(ct_image, rt_channels, unique=True)
    for _, dwell_position in enumerate(dwell_positions):
        ct_slices.append(get_slice_by_dwell(ct_array, dwell_position, axis=0))
        dose_slice_tmp = get_slice_by_dwell(dose_array, dwell_position, axis=0)
        dose_slices.append(dose_slice_tmp)
        dwell_candidates = extract_peak_dose_points(dose_slice_tmp)
        mask_slices.append(get_slice_by_dwell(mask_array, dwell_position, axis=0))

    ct_slices = np.array(ct_slices)
    dose_slices = np.array(dose_slices)
    mask_slices = np.array(mask_slices)

    return ct_slices, dose_slices, mask_slices, dwell_positions, dwell_candidates

def get_slice_by_dwell(ct_array: np.ndarray, dwell_position: np.ndarray, axis: int = 0):
    """
    Extracts a 2D slice from a 3D CT array at the position of a dwell point along a specified axis.
    axis: 0 for axial, 1 for coronal, 2 for sagittal.
    """
    slice_index_rounded = int(np.clip(np.round(dwell_position[-1+axis]), 0, ct_array.shape[axis] - 1))
    slicer = [slice(None)] * 3
    slicer[axis] = slice_index_rounded
    return ct_array[tuple(slicer)]

def map_channel_dwell_indices(channels, dwell_points):
    """Return flattened dwell indices for every RTPLAN channel."""

    mapping = []
    dwell_idx = 0
    for channel in channels:
        channel_indices = []
        for pos in channel.positions_cm:
            if pos is None:
                continue
            if dwell_idx >= len(dwell_points):
                raise ValueError("Flattened dwell list shorter than RTPLAN topology.")
            channel_indices.append(dwell_idx)
            dwell_idx += 1
        mapping.append(channel_indices)
    if dwell_idx != len(dwell_points):
        raise ValueError("Flattened dwell list longer than RTPLAN topology.")
    return mapping


def _normalise_distribution(values, target_sum):
    weights = np.asarray(values, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if weights.sum() <= 0.0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    return weights * target_sum


def adjust_channel_dwells(
    rt_channels,
    *,
    mode="perturb",
    noise_scale=1.0,
):
    """Return a copy of ``dwell_points`` with channel-wise dwell times adjusted."""

    strengths: List[float] = []
    for channel in rt_channels:
        vals = channel.strengths_U
        if vals.size:
            strengths.extend(float(v) for v in vals if np.isfinite(v) and v > 0.0)
    default_strength_U = float(np.mean(strengths)) if strengths else 30_000.0

    dwells, _ = dosecal.dwells_from_records(rt_channels, default_strength_U=default_strength_U)
    channel_indices = map_channel_dwell_indices(rt_channels, dwells)

    updated = copy.deepcopy(dwells)

    for indices in channel_indices:
        if not indices:
            continue

        base = np.array([dwells[i].dwell_time_s for i in indices], dtype=float)
        total = float(base.sum())
        if total <= 0.0:
            continue

        if mode == "perturb":
            noise = np.random.normal(loc=0.0, scale=noise_scale, size=base.size)
            proposal = base * (1.0 + noise)
        elif mode == "random":
            proposal = np.random.normal(loc=0.0, scale=noise_scale, size=base.size)
        else:
            return rt_channels

        scaled = _normalise_distribution(proposal, total)
        for idx, dwell_idx in enumerate(indices):
            updated[dwell_idx].dwell_time_s = float(scaled[idx])

    dosecal.rebuild_channels(rt_channels, updated)
    return dosecal.rebuild_channels(rt_channels, updated)

def extract_peak_dose_points(
    dose_slice: np.ndarray,
    min_distance: int = 1,
    threshold_rel: float = 0.1
):

    coords = peak_local_max(
        dose_slice,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        exclude_border=False
    )

    values = dose_slice[coords[:, 0], coords[:, 1]]
    order = np.argsort(values)[::-1]
    peaks_yx = coords[order]
    peaks_xy = peaks_yx[:, ::-1]
    return peaks_xy