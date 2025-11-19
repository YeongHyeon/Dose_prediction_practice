"""TG-43 dose calculation helpers driven by RTPLAN dwell data."""

from __future__ import annotations

import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import SimpleITK as sitk

from tg43.logging_utils import get_logger

import tg43.utils as utils
import tg43.contour_helper as chelp

EPSILON = 1e-6
XLSX_NAMESPACE_MAP = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
DEFAULT_DVH_STRUCTURES = ("Bladder", "Rectum", "Sigmoid", "Bowel", "HR-CTV")
logger = get_logger(__name__)

try:
    import tg43.dicom_helper as dhelp  # type: ignore
except ImportError as exc:  # pragma: no cover - utils lives next to this module
    raise ImportError("utils.py must be importable alongside dosecal.py") from exc

from tg43.dicom_helper import ChannelInfo


@dataclass
class TG43TableSet:
    """Tabulated TG-43 kernel components prepared for interpolation.

    Fields hold the anisotropy grid, the radial dose samples, and reference
    constants used when turning source strength into dose rate.
    """
    r_grid_cm: np.ndarray
    theta_grid_rad: np.ndarray
    F_table: np.ndarray  # shape (Ntheta, Nr)
    g_r_cm: np.ndarray
    g_vals: np.ndarray
    lambda_Gy_per_h_U: float = 1.109
    reference_r_cm: float = 1.0

    def gL(self, r_cm: np.ndarray) -> np.ndarray:
        """Evaluate the radial dose function ``g(r)`` inside the tabulated span.

        Distances are clipped to the calibrated range before interpolation so
        the returned array remains aligned with the incoming ordering even when
        callers supply out-of-range values.

        Parameters
        ----------
        r_cm : numpy.ndarray
            Radial distances in centimetres.

        Returns
        -------
        numpy.ndarray
            Interpolated ``g(r)`` values arranged in a flat array.
        """
        r = np.asarray(r_cm, dtype=float)
        r_clip = np.clip(r, self.g_r_cm[0], self.g_r_cm[-1])
        return np.interp(r_clip, self.g_r_cm, self.g_vals)

    def F(self, r_cm: np.ndarray, theta_rad: np.ndarray) -> np.ndarray:
        """Look up the anisotropy function ``F(r, \theta)`` via bilinear blending.

        Both radii and angles are clipped to the supported grids, while polar
        angles wrap over ``pi`` to honour the axial symmetry of the source.

        Parameters
        ----------
        r_cm : numpy.ndarray
            Radial distances in centimetres.
        theta_rad : numpy.ndarray
            Polar angles in radians.

        Returns
        -------
        numpy.ndarray
            Bilinearly interpolated anisotropy values obeying NumPy broadcast
            rules.
        """
        r = np.asarray(r_cm, dtype=float)
        theta = np.asarray(theta_rad, dtype=float)
        r_clip = np.clip(r, self.r_grid_cm[0], self.r_grid_cm[-1])
        theta_mod = np.mod(theta, math.pi)
        theta_clip = np.clip(theta_mod, self.theta_grid_rad[0], self.theta_grid_rad[-1])

        r_idx = np.searchsorted(self.r_grid_cm, r_clip, side="right") - 1
        r_idx = np.clip(r_idx, 0, len(self.r_grid_cm) - 2)
        t_idx = np.searchsorted(self.theta_grid_rad, theta_clip, side="right") - 1
        t_idx = np.clip(t_idx, 0, len(self.theta_grid_rad) - 2)

        r0 = self.r_grid_cm[r_idx]
        r1 = self.r_grid_cm[r_idx + 1]
        t0 = self.theta_grid_rad[t_idx]
        t1 = self.theta_grid_rad[t_idx + 1]

        denom_r = np.where(r1 > r0, r1 - r0, 1.0)
        denom_t = np.where(t1 > t0, t1 - t0, 1.0)

        fr = (r_clip - r0) / denom_r
        ft = (theta_clip - t0) / denom_t

        f00 = self.F_table[t_idx, r_idx]
        f01 = self.F_table[t_idx, r_idx + 1]
        f10 = self.F_table[t_idx + 1, r_idx]
        f11 = self.F_table[t_idx + 1, r_idx + 1]

        return (
            (1.0 - fr) * (1.0 - ft) * f00
            + fr * (1.0 - ft) * f01
            + (1.0 - fr) * ft * f10
            + fr * ft * f11
        )


@dataclass
class DwellPoint:
    """Single brachytherapy dwell point expressed in centimetres and seconds.

    ``axis`` stores the unit vector that aligns the source with respect to the
    voxel grid, enabling anisotropy lookups during kernel evaluation.
    """
    position_cm: np.ndarray
    axis: np.ndarray
    dwell_time_s: float
    source_strength_U: float


@dataclass
class RectilinearGrid:
    """Axis-aligned rectilinear grid whose node spacing/origin use centimetres."""
    origin_cm: np.ndarray
    spacing_cm: np.ndarray
    shape: Tuple[int, int, int]


@dataclass
class DoseComputationResult:
    """Bundle capturing coarse TG-43 dose data and CT-resampled derivatives.

    The metadata dictionary aggregates dwell statistics, grid spacing, and
    kernel bookkeeping that are helpful for QA and reporting.
    """
    coarse_grid: RectilinearGrid
    coarse_volume: np.ndarray
    coarse_image: sitk.Image
    resampled_image: sitk.Image
    resampled_array: np.ndarray
    metadata: Dict[str, float]


def _image_bounds_cm(image: sitk.Image) -> Tuple[np.ndarray, np.ndarray]:
    """Return the axis-aligned bounds of ``image`` expressed in centimetres.

    Degenerate images (including single-voxel volumes) fall back to using the
    physical origin so downstream grid logic always receives a valid box.
    """

    size = image.GetSize()
    spacing_cm = np.array(image.GetSpacing(), dtype=float) / 10.0
    corners = []
    for i in (0, max(size[0] - 1, 0)):
        for j in (0, max(size[1] - 1, 0)):
            for k in (0, max(size[2] - 1, 0)):
                phys_mm = np.array(image.TransformIndexToPhysicalPoint((i, j, k)), dtype=float)
                corners.append(phys_mm / 10.0)
    if not corners:
        phys_mm = np.array(image.GetOrigin(), dtype=float)
        corners = [phys_mm / 10.0]
    corners = np.vstack(corners)
    min_corner = corners.min(axis=0) - 0.5 * spacing_cm
    max_corner = corners.max(axis=0) + 0.5 * spacing_cm
    return min_corner, max_corner


def _xlsx_column_index(cell_ref: str) -> int:
    """Translate an Excel cell reference into a zero-based column offset.

    Alphabetic characters are interpreted case-insensitively so references like
    ``"C5"`` and ``"c5"`` resolve to the same positional index.
    """
    col = "".join(ch for ch in cell_ref if ch.isalpha())
    idx = 0
    for ch in col:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1


def _load_xlsx_numeric_table(path: Union[str, Path]) -> List[List[float]]:
    """Load the first worksheet of an XLSX workbook into a float matrix.

    Missing cells are represented as ``math.nan`` so downstream interpolation
    code can detect gaps without additional sentinels.

    Parameters
    ----------
    path : str or Path
        Location of the workbook on disk.

    Returns
    -------
    list[list[float]]
        Row-major float values with ``math.nan`` placeholders.

    Raises
    ------
    FileNotFoundError
        If the workbook cannot be located.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"XLSX file not found: {path}")

    with zipfile.ZipFile(path) as zf:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall(".//s:si", XLSX_NAMESPACE_MAP):
                text = "".join(t.text or "" for t in si.findall(".//s:t", XLSX_NAMESPACE_MAP))
                shared_strings.append(text)

        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))

    table: List[List[float]] = []
    for row in sheet.findall(".//s:sheetData/s:row", XLSX_NAMESPACE_MAP):
        cells: Dict[int, float] = {}
        for cell in row.findall("s:c", XLSX_NAMESPACE_MAP):
            ref = cell.get("r")
            if ref is None:
                continue
            col_idx = _xlsx_column_index(ref)
            text_type = cell.get("t")
            value_node = cell.find("s:v", XLSX_NAMESPACE_MAP)
            if value_node is None:
                cells[col_idx] = math.nan
                continue
            raw = value_node.text or ""
            if text_type == "s":
                try:
                    raw = shared_strings[int(raw)]
                except (IndexError, ValueError):
                    raw = ""
            try:
                val = float(raw)
            except ValueError:
                val = math.nan
            cells[col_idx] = val
        if cells:
            row_vals = [math.nan] * (max(cells) + 1)
            for idx, val in cells.items():
                row_vals[idx] = val
            table.append(row_vals)
    return table


def load_nucletron_tg43_tables(
    anisotropy_path: Union[str, Path],
    radial_path: Union[str, Path],
    *,
    lambda_Gy_per_h_U: float = 1.109,
    reference_r_cm: float = 1.0,

) -> TG43TableSet:
    """Parse TG-43 anisotropy and radial dose tables from ESTRO workbooks.

    The loader extracts a clean radial grid, resolves duplicate radii, and
    converts both tables into NumPy arrays that can be interpolated efficiently
    during dose evaluation.

    Parameters
    ----------
    anisotropy_path : str or Path
        Workbook containing the anisotropy function values ``F(r, \theta)``.
    radial_path : str or Path
        Workbook containing the radial dose function ``g(r)``.
    lambda_Gy_per_h_U : float, optional
        Dose-rate constant used to convert strength to Gy/h/U.
    reference_r_cm : float, optional
        Reference radius (in cm) for the geometry factor.

    Returns
    -------
    TG43TableSet
        Interpolation-ready table bundle for subsequent computations.

    Raises
    ------
    ValueError
        If mandatory table content is missing or malformed.
    """

    # Parse anisotropy workbook into polar grid samples
    F_rows = _load_xlsx_numeric_table(anisotropy_path)
    if not F_rows:
        raise ValueError("Anisotropy table is empty")

    r_candidates = [val for val in F_rows[0][1:] if not math.isnan(val)]
    if not r_candidates:
        raise ValueError("Could not parse radial grid from anisotropy file")
    r_grid_raw = np.array(r_candidates, dtype=float)
    r_grid_cm, unique_idx = np.unique(r_grid_raw, return_index=True)

    theta_vals_deg: List[float] = []
    F_values: List[List[float]] = []
    for row in F_rows[1:]:
        if not row or math.isnan(row[0]):
            continue
        theta_vals_deg.append(row[0])
        row_vals: List[float] = []
        for idx in unique_idx:
            col = idx + 1
            if col < len(row) and not math.isnan(row[col]):
                row_vals.append(row[col])
            else:
                row_vals.append(row_vals[-1] if row_vals else 1.0)
        F_values.append(row_vals)

    if not F_values:
        raise ValueError("No anisotropy data rows detected")

    theta_grid_rad = np.deg2rad(np.array(theta_vals_deg, dtype=float))
    F_table = np.array(F_values, dtype=float)

    # Parse radial dose workbook into monotonic samples
    g_rows = _load_xlsx_numeric_table(radial_path)
    r_g_list: List[float] = []
    g_list: List[float] = []
    for row in g_rows:
        if len(row) < 2:
            continue
        r_val, g_val = row[0], row[1]
        if math.isnan(r_val) or math.isnan(g_val):
            continue
        r_g_list.append(r_val)
        g_list.append(g_val)
    if not r_g_list:
        raise ValueError("No radial dose function entries detected")

    r_g_cm = np.array(r_g_list, dtype=float)
    g_vals = np.array(g_list, dtype=float)
    order = np.argsort(r_g_cm)
    r_g_cm = r_g_cm[order]
    g_vals = g_vals[order]

    return TG43TableSet(
        r_grid_cm=r_grid_cm,
        theta_grid_rad=theta_grid_rad,
        F_table=F_table,
        g_r_cm=r_g_cm,
        g_vals=g_vals,
        lambda_Gy_per_h_U=lambda_Gy_per_h_U,
        reference_r_cm=reference_r_cm,
    )


def _estimate_axes(points_cm: np.ndarray) -> np.ndarray:
    """Estimate unit tangents for dwell positions along a catheter track.

    Central differences provide smooth axes for interior dwells, while the
    endpoints fall back to forward/backward differences to keep orientations
    well-defined. When a segment collapses numerical noise, the axis defaults
    to the positive ``z`` direction.

    Parameters
    ----------
    points_cm : numpy.ndarray
        Array of shape ``(N, 3)`` with dwell positions in centimetres.

    Returns
    -------
    numpy.ndarray
        Array of unit vectors aligned with each dwell.
    """
    n = len(points_cm)
    if n == 0:
        return np.zeros((0, 3), dtype=float)
    axes = np.zeros_like(points_cm)
    if n == 1:
        axes[0] = np.array([0.0, 0.0, 1.0])
        return axes
    for idx in range(n):
        if idx == 0:
            diff = points_cm[1] - points_cm[0]
        elif idx == n - 1:
            diff = points_cm[-1] - points_cm[-2]
        else:
            diff = points_cm[idx + 1] - points_cm[idx - 1]
        norm = np.linalg.norm(diff)
        axes[idx] = diff / norm if norm > EPSILON else np.array([0.0, 0.0, 1.0])
    return axes


def dwells_from_records(
    records: Sequence[ChannelInfo],
    *,
    default_strength_U: float = 30_000.0,
) -> Tuple[List[DwellPoint], Dict[str, float]]:
    """Flatten RTPLAN channel information into per-dwell entries and metadata.

    Channel totals, timing overrides, and missing strengths are harmonised so
    the returned dwell list is ready for kernel evaluation, while the metadata
    captures aggregate dwell counts, total dwell time, and the number of
    channels that relied on relative timing.
    """

    channels = list(records)
    if not channels:
        return [], {"num_channels": 0, "num_dwells": 0, "total_dwell_time_s": 0.0, "relative_time_channels": 0}

    dwell_points: List[DwellPoint] = []
    total_time = 0.0
    relative_channels = 0

    for channel in channels:
        weights = np.asarray(channel.cumulative_weights, dtype=float)
        if weights.size == 0:
            continue

        total_time_s = channel.total_time_s
        final_weight = channel.final_cumulative_weight
        if final_weight is None or final_weight <= 0.0:
            final_weight = float(weights[-1]) if weights.size else 0.0

        use_channel_total = (
            total_time_s is not None and total_time_s > 0.0 and final_weight and final_weight > 0.0
        )
        if total_time_s is None and final_weight and 0.0 < final_weight <= 1.05:
            relative_channels += 1

        scale = (total_time_s / final_weight) if use_channel_total else 1.0

        strengths = np.asarray(channel.strengths_U, dtype=float)

        prev_weight = 0.0
        pending = 0.0
        channel_positions: List[np.ndarray] = []
        dwell_indices: List[int] = []

        for idx, weight in enumerate(weights):
            delta = max(weight - prev_weight, 0.0)
            prev_weight = weight
            pending += delta

            pos = channel.positions_cm[idx] if idx < len(channel.positions_cm) else None
            if pos is None:
                continue

            pos_cm = np.asarray(pos, dtype=float)
            strength = strengths[idx] if idx < strengths.size else np.nan
            if not np.isfinite(strength) or strength <= 0.0:
                strength = default_strength_U

            dwell_time = pending * scale
            pending = 0.0

            dwell_points.append(
                DwellPoint(
                    position_cm=pos_cm,
                    axis=np.zeros(3, dtype=float),
                    dwell_time_s=float(dwell_time),
                    source_strength_U=float(strength),
                )
            )
            channel_positions.append(pos_cm)
            dwell_indices.append(len(dwell_points) - 1)

        if not dwell_indices:
            continue

        if pending > 0.0:
            dwell_points[dwell_indices[-1]].dwell_time_s += pending * scale

        channel_sum = sum(dwell_points[i].dwell_time_s for i in dwell_indices)
        if use_channel_total and channel_sum > 0.0 and total_time_s is not None:
            scale_factor = total_time_s / channel_sum
            for i in dwell_indices:
                dwell_points[i].dwell_time_s *= scale_factor
            channel_sum = total_time_s

        axes = _estimate_axes(np.vstack(channel_positions))
        for i, axis in zip(dwell_indices, axes):
            dwell_points[i].axis = axis.astype(float)

        total_time += channel_sum

    metadata = {
        "num_channels": float(len(channels)),
        "num_dwells": float(len(dwell_points)),
        "total_dwell_time_s": float(total_time),
        "relative_time_channels": float(relative_channels),
    }
    return dwell_points, metadata


def build_rectilinear_grid(
    dwells: Sequence[DwellPoint],
    *,
    spacing_mm: float = 2.5,
    margin_mm: float = 20.0,
    bounding_image: Optional[sitk.Image] = None,
    max_distance_cm: Optional[float] = None,
) -> RectilinearGrid:
    """Construct a rectilinear grid that bounds the supplied dwell positions.

    The grid centres voxels around the dwell cloud, supports scalar or
    per-axis spacings, and expands to include an optional reference image so
    downstream resampling captures both high- and low-dose regions.

    Parameters
    ----------
    dwells : Sequence[DwellPoint]
        Dwell points to enclose.
    spacing_mm : float or sequence of float, optional
        Isotropic spacing or a three-component per-axis spacing (millimetres).
    margin_mm : float, optional
        Margin added beyond the dwell extents (millimetres).
    bounding_image : SimpleITK.Image, optional
        Image whose physical bounds should also lie inside the grid.
    max_distance_cm : float, optional
        When provided, restricts the bounding image coverage to the dwell extent
        plus this radius to avoid evaluating empty far-field voxels.

    Returns
    -------
    RectilinearGrid
        Grid definition expressed in centimetres.

    Raises
    ------
    ValueError
        If ``dwells`` is empty.
    """
    if not dwells:
        raise ValueError("No dwell points provided for grid construction")
    positions = np.vstack([d.position_cm for d in dwells])
    
    if np.isscalar(spacing_mm) or len(spacing_mm) == 1:
        spacing_cm = np.full(3, float(spacing_mm) / 10.0, dtype=float)
    else:
        spacing_cm = np.array(spacing_mm, dtype=float) / 10.0
    
    margin_cm = float(margin_mm) / 10.0
    dwell_min = positions.min(axis=0)
    dwell_max = positions.max(axis=0)
    min_corner = dwell_min - margin_cm
    max_corner = dwell_max + margin_cm

    if bounding_image is not None:
        img_min, img_max = _image_bounds_cm(bounding_image)
        if max_distance_cm is not None:
            limit_min = dwell_min - max_distance_cm
            limit_max = dwell_max + max_distance_cm
            img_min = np.maximum(img_min, limit_min)
            img_max = np.minimum(img_max, limit_max)
        if np.all(img_min < img_max):
            min_corner = np.minimum(min_corner, img_min)
            max_corner = np.maximum(max_corner, img_max)

    extents = np.maximum(max_corner - min_corner, spacing_cm * 2.0)
    shape = np.maximum(1, np.ceil(extents / spacing_cm).astype(int) + 1)
    return RectilinearGrid(
        origin_cm=min_corner.astype(float),
        spacing_cm=spacing_cm,
        shape=(int(shape[0]), int(shape[1]), int(shape[2])),
    )


def _grid_coordinates(grid: RectilinearGrid) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return voxel-centre coordinates in both flat and structured layouts.

    The helper exposes a flattened view for vectorised calculations alongside
    ``meshgrid`` outputs that preserve the original grid topology.

    Parameters
    ----------
    grid : RectilinearGrid
        Grid definition including origin, spacing, and shape.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Flattened coordinates followed by 3D meshgrids (X, Y, Z).
    """
    nx, ny, nz = grid.shape
    xs = grid.origin_cm[0] + (np.arange(nx, dtype=float) + 0.5) * grid.spacing_cm[0]
    ys = grid.origin_cm[1] + (np.arange(ny, dtype=float) + 0.5) * grid.spacing_cm[1]
    zs = grid.origin_cm[2] + (np.arange(nz, dtype=float) + 0.5) * grid.spacing_cm[2]
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # use ravel to avoid unnecessary copies
    return coords, X, Y, Z


def compute_tg43_dose_on_grid(
    dwells: Sequence[DwellPoint],
    grid: RectilinearGrid,
    tables: TG43TableSet,
    *,
    max_distance_cm: Optional[float] = 10.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Evaluate TG-43 dose contributions across the provided rectilinear grid.

    Each dwell applies the TG-43 kernel to voxel-centre coordinates, with an
    optional distance cut-off to trim far-field evaluations. Aggregate dwell
    strength-time products are tracked for later reporting.

    Parameters
    ----------
    dwells : Sequence[DwellPoint]
        Dwell points with timing and source strength data.
    grid : RectilinearGrid
        Grid on which the dose is to be evaluated.
    tables : TG43TableSet
        Interpolation tables for anisotropy and radial dose functions.
    max_distance_cm : float, optional
        Optional cut-off radius (centimetres) that trims inactive voxels.

    Returns
    -------
    tuple[numpy.ndarray, dict]
        Dose volume in Gy and metadata with dwell counts/strength-time totals.
    """
    voxel_positions_cm, _, _, _ = _grid_coordinates(grid)
    dose_Gy_flat = np.zeros(voxel_positions_cm.shape[0], dtype=np.float32)
    cumulative_strength_time_U_s = 0.0
    lambda_Gy_per_s_per_U = tables.lambda_Gy_per_h_U / 3600.0

    for dwell in dwells:
        strength_time_U_s = dwell.source_strength_U * dwell.dwell_time_s  # Sk × t
        if strength_time_U_s <= 0.0:
            continue
        cumulative_strength_time_U_s += strength_time_U_s

        displacement_cm = voxel_positions_cm - dwell.position_cm[np.newaxis, :]
        radial_distance_cm = np.linalg.norm(displacement_cm, axis=1)

        if max_distance_cm is not None:  # apply cut-off mask to reduce computation
            active_mask = radial_distance_cm <= max_distance_cm
            if not np.any(active_mask):
                continue
            displacement_active_cm = displacement_cm[active_mask]
            radial_distance_active_cm = np.maximum(radial_distance_cm[active_mask], EPSILON)
        else:
            active_mask = slice(None)
            displacement_active_cm = displacement_cm
            radial_distance_active_cm = np.maximum(radial_distance_cm, EPSILON)

        source_axis = dwell.axis
        axis_norm = np.linalg.norm(source_axis)
        if axis_norm <= EPSILON:
            source_axis = np.array([0.0, 0.0, 1.0])
        else:
            source_axis = source_axis / axis_norm

        cos_theta = (displacement_active_cm @ source_axis) / np.maximum(radial_distance_active_cm, EPSILON)
        theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        # TG-43: Ḋ(r, θ) = Λ · Sk · G(r,θ)/G(r₀,θ₀) · g(r) · F(r,θ)
        geometry_ratio = (1/radial_distance_active_cm**2) / (1/tables.reference_r_cm**2)
        radial_dose_factor = tables.gL(radial_distance_active_cm) # g(r)
        anisotropy_factor = tables.F(radial_distance_active_cm, theta_rad) # F(r, θ)
        dose_rate_kernel = lambda_Gy_per_s_per_U * geometry_ratio * radial_dose_factor * anisotropy_factor
        dwell_dose_Gy = (dose_rate_kernel * strength_time_U_s).astype(np.float32)

        if isinstance(active_mask, slice):
            dose_Gy_flat += dwell_dose_Gy
        else:
            dose_Gy_flat[active_mask] += dwell_dose_Gy

    dose_volume = dose_Gy_flat.reshape(grid.shape)
    metadata = {
        "num_dwells": float(len(dwells)),
        "total_dwell_weight_U_s": float(cumulative_strength_time_U_s),
    }
    return dose_volume, metadata

def compute_tg43_dose_at_points(
    dwells: Sequence[DwellPoint],
    points_cm: np.ndarray,
    tables: TG43TableSet,
    *,
    max_distance_cm: Optional[float] = None,
    dwell_time_override_s: Optional[float] = None,
) -> np.ndarray:
    """Evaluate TG-43 dose contributions at arbitrary reference points.

    The computation mirrors :func:`compute_tg43_dose_on_grid` but restricts the
    evaluation to the supplied physical coordinates. This is useful when
    probing classical Point-A/B references, or when solving for dwell times
    that yield a prescribed dose at a handful of QA markers.

    Parameters
    ----------
    dwells : Sequence[DwellPoint]
        Dwell points with timing and source strength data.
    points_cm : array-like
        Array of shape ``(N, 3)`` containing reference point coordinates in
        centimetres.
    tables : TG43TableSet
        Interpolation tables for anisotropy and radial dose functions.
    max_distance_cm : float, optional
        Optional cut-off that skips dwells further than ``max_distance_cm`` away
        from every reference point.
    dwell_time_override_s : float, optional
        When provided, each dwell uses this uniform dwell time instead of the
        value stored in ``dwell_time_s``. Passing ``1.0`` is a convenient way
        to retrieve dose-per-second rates.

    Returns
    -------
    numpy.ndarray
        Dose in Gy for each reference point after accounting for the dwell
        times (or override) and source strengths.
    """
    pts = np.asarray(points_cm, dtype=float)
    if pts.ndim == 1:
        if pts.size != 3:
            raise ValueError("points_cm must have length 3 when passing a single reference point.")
        pts = pts[np.newaxis, :]
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_cm must be shaped (N, 3) in centimetres.")
    if pts.size == 0:
        return np.zeros((0,), dtype=np.float32)

    lambda_Gy_per_s_per_U = tables.lambda_Gy_per_h_U / 3600.0
    override_time = float(dwell_time_override_s) if dwell_time_override_s is not None else None
    dose_Gy = np.zeros(pts.shape[0], dtype=np.float64)

    for dwell in dwells:
        dwell_time = override_time if override_time is not None else dwell.dwell_time_s
        strength_time_U_s = dwell.source_strength_U * dwell_time
        if strength_time_U_s <= 0.0:
            continue

        displacement_cm = pts - dwell.position_cm[np.newaxis, :]
        radial_distance_cm = np.linalg.norm(displacement_cm, axis=1)

        if max_distance_cm is not None:
            active_mask = radial_distance_cm <= max_distance_cm
            if not np.any(active_mask):
                continue
            displacement_active_cm = displacement_cm[active_mask]
            radial_distance_active_cm = np.maximum(radial_distance_cm[active_mask], EPSILON)
        else:
            active_mask = slice(None)
            displacement_active_cm = displacement_cm
            radial_distance_active_cm = np.maximum(radial_distance_cm, EPSILON)

        axis = dwell.axis
        axis_norm = np.linalg.norm(axis)
        if axis_norm <= EPSILON:
            axis = np.array([0.0, 0.0, 1.0])
        else:
            axis = axis / axis_norm

        cos_theta = (displacement_active_cm @ axis) / np.maximum(radial_distance_active_cm, EPSILON)
        theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        geometry_ratio = (1 / radial_distance_active_cm**2) / (1 / tables.reference_r_cm**2)
        radial_dose_factor = tables.gL(radial_distance_active_cm)
        anisotropy_factor = tables.F(radial_distance_active_cm, theta_rad)
        dose_rate_kernel = lambda_Gy_per_s_per_U * geometry_ratio * radial_dose_factor * anisotropy_factor
        dwell_dose_Gy = (dose_rate_kernel * strength_time_U_s).astype(np.float64)

        if isinstance(active_mask, slice):
            dose_Gy += dwell_dose_Gy
        else:
            dose_Gy[active_mask] += dwell_dose_Gy

    return dose_Gy.astype(np.float32)

def dose_volume_to_image(dose_volume: np.ndarray, grid: RectilinearGrid) -> sitk.Image:
    """Convert a NumPy dose volume into a SimpleITK image using grid metadata.

    Axes are reordered into SimpleITK's ``(z, y, x)`` convention and the result
    carries voxel spacing/origin derived from the rectilinear grid.
    """
    arr = dose_volume.astype(np.float32).transpose(2, 1, 0)
    img = sitk.GetImageFromArray(arr)
    spacing_mm = tuple((grid.spacing_cm * 10.0).tolist())
    origin_mm = tuple((grid.origin_cm * 10.0).tolist())
    img.SetSpacing(spacing_mm)
    img.SetOrigin(origin_mm)
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return img


def calculate_tg43_rectilinear_dose(
    channels: Sequence[ChannelInfo],
    anisotropy_path: Union[str, Path],
    radial_path: Union[str, Path],
    *,
    grid_spacing_mm: float = 2.5,
    margin_mm: float = 20.0,
    max_distance_cm: Optional[float] = 10.0,
    default_strength_U: Optional[float] = None,
    bounding_image: Optional[sitk.Image] = None,
) -> Tuple[sitk.Image, np.ndarray, RectilinearGrid, Dict[str, float], List[DwellPoint]]:
    """Compute the TG-43 dose grid and convert it to a SimpleITK image.

    TG-43 tables are loaded from disk, dwell metadata from
    :func:`tg43.dicom_helper.load_rtplan_by_channel` is flattened, and the
    kernel is integrated on a coarse rectilinear grid before being wrapped in a
    SimpleITK image.

    Parameters
    ----------
    channels : Sequence[ChannelInfo]
        RTPLAN dwell information, typically from ``dicom_helper.load_rtplan_by_channel``.
    anisotropy_path : str or Path
        Path to the anisotropy table workbook.
    radial_path : str or Path
        Path to the radial dose function workbook.
    grid_spacing_mm : float, optional
        Spacing of the coarse rectilinear grid (millimetres).
    margin_mm : float, optional
        Margin added to the convex hull of the dwell positions (millimetres).
    max_distance_cm : float, optional
        Optional cut-off radius for dose contributions (centimetres).
    default_strength_U : float, optional
        Fallback air-kerma strength when records omit the value.
    bounding_image : SimpleITK.Image, optional
        Optional image whose extent must be included in the dose grid (for example
        a CT volume when low-dose coverage is required).

    Returns
    -------
    tuple
        ``(dose_image, dose_volume, grid, metadata, dwell_points)`` where
        ``metadata`` captures grid spacing, dwell totals, and kernel summaries.

    Raises
    ------
    ValueError
        If no valid dwell positions are discovered in ``channels``.
    """
    tables = load_nucletron_tg43_tables(anisotropy_path, radial_path)

    if default_strength_U is None:
        strengths: List[float] = []
        for channel in channels:
            vals = channel.strengths_U
            if vals.size:
                strengths.extend(float(v) for v in vals if np.isfinite(v) and v > 0.0)
        default_strength_U = float(np.mean(strengths)) if strengths else 30_000.0

    dwells, dwell_meta = dwells_from_records(channels, default_strength_U=default_strength_U)

    if not dwells:
        raise ValueError("No dwell positions with positive dwell time were found in RTPLAN channels")
    grid = build_rectilinear_grid(
        dwells,
        spacing_mm=grid_spacing_mm,
        margin_mm=margin_mm,
        bounding_image=bounding_image,
        max_distance_cm=max_distance_cm,
    )
    dose_volume, kernel_meta = compute_tg43_dose_on_grid(dwells, grid, tables, max_distance_cm=max_distance_cm)
    dose_image = dose_volume_to_image(dose_volume, grid)
    metadata = {**dwell_meta, **kernel_meta, "grid_spacing_mm": grid_spacing_mm}
    return dose_image, dose_volume, grid, metadata, dwells


def calculate_and_resample_to_ct(
    ct_image: sitk.Image,
    channels: Sequence[ChannelInfo],
    anisotropy_path: Union[str, Path],
    radial_path: Union[str, Path],
    *,
    grid_spacing_mm: float = 2.5,
    margin_mm: float = 20.0,
    max_distance_cm: Optional[float] = 10.0,
    default_strength_U: Optional[float] = None,
) -> DoseComputationResult:
    """Compute TG-43 dose and resample it to the CT reference frame.

    The rectilinear dose grid is generated first, then resampled via
    :func:`tg43.dicom_helper.resample_to_reference` so that the final volume
    aligns exactly with the CT geometry. The coarse grid bounds are expanded to
    include the CT extent, preserving low-dose coverage after resampling.

    Parameters
    ----------
    ct_image : SimpleITK.Image
        Reference CT image that defines the target geometry.
    channels : Sequence[ChannelInfo]
        RTPLAN dwell information.
    anisotropy_path : str or Path
        Path to the anisotropy table workbook.
    radial_path : str or Path
        Path to the radial dose function workbook.
    grid_spacing_mm : float, optional
        Spacing of the intermediate coarse grid (millimetres).
    margin_mm : float, optional
        Margin applied when building the coarse grid (millimetres).
    max_distance_cm : float, optional
        Optional maximum distance (centimetres) for dose contribution.
    default_strength_U : float, optional
        Fallback strength used when omitted in the records.

    Returns
    -------
    DoseComputationResult
        Aggregated output with coarse and CT-aligned dose volumes.
    """
    coarse_image, coarse_volume, grid, metadata, _ = calculate_tg43_rectilinear_dose(
        channels,
        anisotropy_path,
        radial_path,
        grid_spacing_mm=grid_spacing_mm,
        margin_mm=margin_mm,
        max_distance_cm=max_distance_cm,
        default_strength_U=default_strength_U,
        bounding_image=ct_image,
    )
    resampled_image, resampled_array, _ = dhelp.resample_to_reference(coarse_image, ct_image)
    metadata["resampled_shape"] = tuple(int(dim) for dim in resampled_array.shape)
    return DoseComputationResult(
        coarse_grid=grid,
        coarse_volume=coarse_volume,
        coarse_image=coarse_image,
        resampled_image=resampled_image,
        resampled_array=resampled_array,
        metadata=metadata,
    )

def rebuild_channels(
    template_channels: Sequence[ChannelInfo],
    optimized_dwells: Sequence[DwellPoint],
) -> List[ChannelInfo]:
    """Distribute optimised dwell times back into the RTPLAN channel layout.

    Parameters
    ----------
    template_channels : Sequence[ChannelInfo]
        Original RTPLAN channel descriptors preserving catheter geometry.
    optimized_dwells : Sequence[DwellPoint]
        Dwell sequence returned by the optimiser.

    Returns
    -------
    list[ChannelInfo]
        Channel records mirroring the RTPLAN layout with updated dwell times.

    Raises
    ------
    ValueError
        If the optimised dwell sequence does not align with the channel topology.
    """

    dwells_iter = iter(optimized_dwells)
    rebuilt = []

    for ch in template_channels:
        positions = [None if p is None else np.asarray(p, float) for p in ch.positions_cm]
        weights = []
        acc = 0.0
        for pos in positions:
            if pos is not None:
                try:
                    acc += float(next(dwells_iter).dwell_time_s)
                except StopIteration:
                    raise ValueError("optimized_dwells shorter than RTPLAN dwell sequence.")
            weights.append(acc)

        final = float(weights[-1]) if weights else 0.0
        rebuilt.append(
            ChannelInfo(
                setup_number=ch.setup_number,
                channel_number=ch.channel_number,
                channel_id=ch.channel_id,
                total_time_s=final or None,
                final_cumulative_weight=final,
                positions_cm=positions,
                cumulative_weights=np.asarray(weights, float),
                relative_positions=np.asarray(ch.relative_positions, float),
                strengths_U=np.asarray(ch.strengths_U, float),
            )
        )

    if any(True for _ in dwells_iter):
        raise ValueError("optimized_dwells contains extra entries; check dwell ordering.")
    return rebuilt


@dataclass
class DVHCurve:
    """Container for cumulative DVH samples expressed in gray.

    ``volume_cc`` is optional and present only when absolute volume data are
    requested.
    """

    dose_bins_Gy: np.ndarray
    volume_percent: np.ndarray
    volume_cc: Optional[np.ndarray]
    metadata: Dict[str, float]


def calculate_dvh_of_mask(
    dose_volume_Gy: np.ndarray,
    structure_mask: np.ndarray,
    *,
    bin_width_Gy: float = 0.1,
    bin_edges_Gy: Optional[Sequence[float]] = None,
    voxel_spacing_mm: Optional[Tuple[float, float, float]] = None,
    dvh_max_dose_Gy: float = 100.0,
) -> DVHCurve:
    """Compute a cumulative DVH from a dose grid and structure mask.

    The routine supports explicit bin edges, optional absolute volume output,
    and records common summary statistics in the returned metadata field. When
    implicit binning is requested a fixed ``0→100 Gy`` range is used so repeated
    calls across different structures always share identical bins, with an
    additional overflow bin capturing any values above the ceiling. Adjust the
    constant ``dvh_max_dose_Gy`` if a different limit is required.

    Parameters
    ----------
    dose_volume_Gy : numpy.ndarray
        Dose grid in gray. Shape must match ``structure_mask``.
    structure_mask : numpy.ndarray
        Boolean mask selecting voxels belonging to the structure of interest.
    bin_width_Gy : float, optional
        Uniform bin width (Gy) used when ``bin_edges_Gy`` is not provided.
    bin_edges_Gy : sequence of float, optional
        Explicit dose bin edges (Gy). Overrides ``bin_width_Gy`` when given.
    voxel_spacing_mm : tuple[float, float, float], optional
        Physical spacing in millimetres used to report absolute volume in cc.

    Returns
    -------
    DVHCurve
        Cumulative DVH sampled at the lower edge of each bin.

    Raises
    ------
    ValueError
        If inputs are inconsistent or the mask selects no voxels.
    """

    dose_volume_Gy = np.asarray(dose_volume_Gy, dtype=float)
    structure_mask = np.asarray(structure_mask, dtype=bool)
    if dose_volume_Gy.shape != structure_mask.shape:
        raise ValueError("Dose volume and structure mask must share the same shape")

    selected = dose_volume_Gy[structure_mask]
    if selected.size == 0:
        raise ValueError("Structure mask does not contain any voxels")

    if bin_edges_Gy is not None:
        edges = np.asarray(bin_edges_Gy, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("bin_edges_Gy must contain at least two values")
        if not np.all(np.diff(edges) > 0):
            raise ValueError("bin_edges_Gy must be strictly increasing")
    else:
        if bin_width_Gy <= 0:
            raise ValueError("bin_width_Gy must be positive")
        max_dose = dvh_max_dose_Gy
        edges = np.arange(0.0, max_dose + bin_width_Gy, bin_width_Gy, dtype=float)
        if edges[-1] < max_dose:
            edges = np.append(edges, edges[-1] + bin_width_Gy)
        if edges.size < 2:
            edges = np.append(edges, edges[-1] + bin_width_Gy)
        # Append an overflow bin so doses beyond the fixed ceiling are captured.
        if not np.isinf(edges[-1]):
            edges = np.append(edges, np.inf)

    hist, edges = np.histogram(selected, bins=edges)
    cumulative_counts = hist[::-1].cumsum()[::-1].astype(float)
    total_voxels = float(cumulative_counts[0]) if cumulative_counts.size else float(selected.size)
    volume_percent = (cumulative_counts / total_voxels) * 100.0

    volume_cc = None
    voxel_volume_mm3 = None
    if voxel_spacing_mm is not None:
        if len(voxel_spacing_mm) != 3:
            raise ValueError("voxel_spacing_mm must contain three components (dx, dy, dz)")
        voxel_volume_mm3 = float(voxel_spacing_mm[0]) * float(voxel_spacing_mm[1]) * float(voxel_spacing_mm[2])
        voxel_volume_cc = voxel_volume_mm3 / 1000.0
        volume_cc = cumulative_counts * voxel_volume_cc

    metadata: Dict[str, float] = {
        "num_voxels": float(total_voxels),
        "min_dose_Gy": float(selected.min()),
        "max_dose_Gy": float(selected.max()),
        "mean_dose_Gy": float(selected.mean()),
    }
    if voxel_volume_mm3 is not None:
        metadata["voxel_volume_mm3"] = voxel_volume_mm3
        metadata["total_volume_cc"] = float(total_voxels) * (voxel_volume_mm3 / 1000.0)

    return DVHCurve(
        dose_bins_Gy=edges[:-1].astype(np.float32),
        volume_percent=volume_percent.astype(np.float32),
        volume_cc=None if volume_cc is None else volume_cc.astype(np.float32),
        metadata=metadata,
    )


def calculate_dvh(
    rtstruct: chelp.RTStructData,
    ct_image: sitk.Image,
    rtplan_result: DoseComputationResult,
    *,
    structures: Sequence[str] | None = None,
) -> Dict[str, DVHCurve]:
    """Compute DVH curves for each requested structure.

    Parameters
    ----------
    rtstruct : RTStructData
        Loaded RTSTRUCT dataset providing structure contours.
    ct_image : SimpleITK.Image
        CT image defining the spatial grid used for rasterisation.
    rtplan_result : DoseComputationResult
        TG-43 dose results resampled to the CT grid.
    structures : Sequence[str], optional
        Collection of structure names to evaluate. Defaults to ``DEFAULT_DVH_STRUCTURES``.

    Returns
    -------
    dict[str, DVHCurve]
        Mapping of structure name to its cumulative DVH curve.
    """

    selected = tuple(structures) if structures is not None else DEFAULT_DVH_STRUCTURES

    dvh_curves: Dict[str, DVHCurve] = {}
    for name in selected:
        mask = chelp.rasterise_structure(rtstruct, name, ct_image)
        if mask is None: 
            logger.warning(f"Structure '{name}' not found in RTSTRUCT; skipping.")
            continue
        if mask.sum() == 0:
            logger.warning(f"Skipping DVH for {name}: mask is empty.")
            continue

        dvh_curves[name] = calculate_dvh_of_mask(
            rtplan_result.resampled_array / 100,
            mask,
            voxel_spacing_mm=rtplan_result.resampled_image.GetSpacing(),
        )

    return dvh_curves


def compare_dvh(
    dvh_baseline: Dict[str, DVHCurve],
    dvh_optimized: Dict[str, DVHCurve],
    *,
    structures: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Summarise per-structure DVH discrepancies.

    Parameters
    ----------
    dvh_baseline : dict[str, DVHCurve]
        Baseline DVHs keyed by structure name.
    dvh_optimized : dict[str, DVHCurve]
        Optimised DVHs keyed by structure name.
    structures : Sequence[str], optional
        Collection of structure names to compare. Defaults to ``DEFAULT_DVH_STRUCTURES``.

    Returns
    -------
    pandas.DataFrame
        Table containing per-structure DVH error statistics.
    """

    errors = {
        "name": [],
        "error_mean_baseline": [],
        "error_std_baseline": [],
        "error_max_baseline": [],
        "error_min_baseline": [],
    }

    selected = tuple(structures) if structures is not None else DEFAULT_DVH_STRUCTURES
    for name in selected:
        if name not in dvh_baseline or name not in dvh_optimized:
            continue
        error = utils.measure_l2_error(
            dvh_baseline[name].volume_percent,
            dvh_optimized[name].volume_percent,
        )
        errors["name"].append(name)
        errors["error_mean_baseline"].append(float(np.mean(error)))
        errors["error_std_baseline"].append(float(np.std(error)))
        errors["error_max_baseline"].append(float(np.max(error)))
        errors["error_min_baseline"].append(float(np.min(error)))

    return pd.DataFrame.from_dict(errors)
