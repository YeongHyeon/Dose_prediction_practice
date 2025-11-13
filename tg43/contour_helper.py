"""Helpers for reading RTSTRUCT datasets and rasterising contours onto CT grids."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pydicom
import SimpleITK as sitk

try:  # matplotlib is used for polygon filling; keep import optional.
    from matplotlib.path import Path as MplPath
except ImportError:  # pragma: no cover - runtime dependency noticed at call site
    MplPath = None


@dataclass
class StructureInfo:
    """ROI metadata (number, name, frame UID) cached from the RTSTRUCT."""

    roi_number: int
    name: str
    frame_of_reference_uid: Optional[str]


@dataclass
class RTStructData:
    """RTSTRUCT dataset plus a convenience mapping of ROI name → StructureInfo."""

    dataset: pydicom.dataset.FileDataset
    structures: Dict[str, StructureInfo]

def load_rtstruct(rtstruct_path: Path | str) -> RTStructData:
    """Load an RTSTRUCT file and build a name-indexed ROI lookup."""

    rtstruct_path = Path(rtstruct_path)
    if not rtstruct_path.exists():
        raise FileNotFoundError(f"RTSTRUCT file not found: {rtstruct_path}")

    ds = pydicom.dcmread(str(rtstruct_path))
    roi_sequence = getattr(ds, "StructureSetROISequence", None)
    if not roi_sequence:
        raise ValueError("RTSTRUCT dataset does not define any StructureSetROISequence entries")

    structures: Dict[str, StructureInfo] = {}
    for roi in roi_sequence:
        number = int(getattr(roi, "ROINumber"))
        name = str(getattr(roi, "ROIName", f"ROI_{number}"))
        frame_uid = getattr(roi, "ReferencedFrameOfReferenceUID", None)
        structures[name] = StructureInfo(
            roi_number=number,
            name=name,
            frame_of_reference_uid=str(frame_uid) if frame_uid is not None else None,
        )

    return RTStructData(dataset=ds, structures=structures)


def available_structures(rtstruct: RTStructData) -> List[str]:
    """Return ROI names in insertion order."""

    return list(rtstruct.structures.keys())


def _roi_contour_sequence(
    dataset: pydicom.dataset.FileDataset,
    roi_number: int,
) -> Optional[pydicom.dataset.Dataset]:
    """Return the contour sequence referencing ``roi_number`` (if present)."""

    for roi_contour in getattr(dataset, "ROIContourSequence", []):
        if int(getattr(roi_contour, "ReferencedROINumber", -1)) == roi_number:
            return roi_contour
    return None


def _continuous_indices(
    image: sitk.Image,
    points_mm: np.ndarray,
) -> np.ndarray:
    """Convert millimetre contour points into fractional image indices."""

    to_index = image.TransformPhysicalPointToContinuousIndex
    return np.asarray([to_index(tuple(point.tolist())) for point in points_mm], dtype=float)


def _fill_polygon(indices_xy: np.ndarray, slice_shape: Tuple[int, int]) -> np.ndarray:
    """Rasterise an index-space polygon into a boolean slice mask via matplotlib."""

    if MplPath is None:
        raise RuntimeError(
            "matplotlib is required to rasterise RTSTRUCT polygons. Install matplotlib or "
            "provide a custom polygon fill implementation."
        )

    path = MplPath(indices_xy)
    xmin = max(int(np.floor(indices_xy[:, 0].min())), 0)
    xmax = min(int(np.ceil(indices_xy[:, 0].max())), slice_shape[1] - 1)
    ymin = max(int(np.floor(indices_xy[:, 1].min())), 0)
    ymax = min(int(np.ceil(indices_xy[:, 1].max())), slice_shape[0] - 1)

    if xmin > xmax or ymin > ymax:
        return np.zeros(slice_shape, dtype=bool)

    grid_x, grid_y = np.meshgrid(
        np.arange(xmin, xmax + 1, dtype=float),
        np.arange(ymin, ymax + 1, dtype=float),
        indexing="xy",
    )
    sample_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    inside = path.contains_points(sample_points, radius=-1e-10)
    mask = np.zeros(slice_shape, dtype=bool)
    mask[ymin : ymax + 1, xmin : xmax + 1] = inside.reshape(grid_y.shape)
    return mask


def rasterise_structure(
    rtstruct: RTStructData,
    structure_name: str,
    reference_image: sitk.Image,
    *,
    roi_numbers: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """Rasterise the requested ROI(s) onto ``reference_image`` and return a boolean mask."""

    size_x, size_y, size_z = reference_image.GetSize()
    mask = np.zeros((size_z, size_y, size_x), dtype=bool)

    if roi_numbers is None:
        roi_info = rtstruct.structures.get(structure_name)
        if roi_info is None:
            alt_name = structure_name.replace("-", "_")
            roi_info = rtstruct.structures.get(alt_name)
            if roi_info is None:
                alt_name = f"{structure_name}c"
                roi_info = rtstruct.structures.get(alt_name)
        if roi_info is None:
            return None
        roi_numbers = [roi_info.roi_number]

    for roi_number in roi_numbers:
        roi_contour = _roi_contour_sequence(rtstruct.dataset, roi_number)
        if roi_contour is None:
            continue

        for contour in getattr(roi_contour, "ContourSequence", []):
            if contour.ContourGeometricType not in (None, "CLOSED_PLANAR"):
                continue
            data = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
            if data.size == 0:
                continue

            indices = _continuous_indices(reference_image, data)
            slice_index = int(round(indices[:, 2].mean()))
            if not (0 <= slice_index < size_z):
                continue

            slice_shape = (size_y, size_x)
            slice_mask = _fill_polygon(indices[:, :2], slice_shape)
            mask[slice_index] |= slice_mask

    return mask


def mask_to_image(mask: np.ndarray, reference_image: sitk.Image, *, value: int = 1) -> sitk.Image:
    """Wrap a ``(z, y, x)`` mask in a SimpleITK image that copies the reference geometry."""

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != tuple(reversed(reference_image.GetSize())):
        raise ValueError("Mask shape does not match reference image size")

    image = sitk.GetImageFromArray(mask.astype(np.uint8) * int(value))
    image.CopyInformation(reference_image)
    return image
