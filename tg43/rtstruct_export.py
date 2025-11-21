"""Utilities for embedding derived masks into RTSTRUCT datasets."""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import UID, generate_uid
import SimpleITK as sitk
from skimage import measure

from tg43.logging_utils import get_logger
import tg43.contour_helper as chelp

logger = get_logger(__name__)


SliceUIDMap = Dict[int, Tuple[str, str]]


def map_ct_slice_uids(
    ct_dir: Path | str,
    *,
    reference_image: sitk.Image,
    series_uid: str | None = None,
) -> SliceUIDMap:
    """Map CT slice indices (k-axis in NumPy array space) to SOPInstanceUIDs."""

    ct_dir = Path(ct_dir)
    if not ct_dir.exists():
        raise FileNotFoundError(f"CT directory not found: {ct_dir}")

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(ct_dir))
    if not series_ids:
        raise RuntimeError(f"No CT series found in {ct_dir}")

    chosen_uid = series_uid if series_uid is not None else series_ids[0]
    if chosen_uid not in series_ids:
        raise ValueError(f"Requested CT series UID {chosen_uid} is unavailable in {ct_dir}")

    file_names = reader.GetGDCMSeriesFileNames(str(ct_dir), chosen_uid)
    if not file_names:
        raise RuntimeError(f"Series {chosen_uid} in {ct_dir} does not contain readable slices.")

    slice_map: SliceUIDMap = {}
    to_index = reference_image.TransformPhysicalPointToContinuousIndex
    for path in file_names:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is None:
            logger.warning("Skipping CT slice %s without ImagePositionPatient.", path)
            continue
        idx = to_index(tuple(float(v) for v in ipp))
        z_index = int(round(idx[2]))
        sop_instance = str(getattr(ds, "SOPInstanceUID"))
        sop_class = str(getattr(ds, "SOPClassUID", UID("1.2.840.10008.5.1.4.1.1.2")))
        slice_map[z_index] = (sop_instance, sop_class)

    if not slice_map:
        raise RuntimeError("CT slice to UID mapping is empty; check the CT inputs.")
    return slice_map


def mask_to_contours(
    mask: np.ndarray,
    *,
    reference_image: sitk.Image,
) -> List[Tuple[int, List[float], int]]:
    """Convert a boolean mask on the CT grid into closed planar contour data."""

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("Mask must be a (z, y, x) volume.")

    expected_shape = tuple(reversed(reference_image.GetSize()))
    if mask.shape != expected_shape:
        raise ValueError(f"Mask shape {mask.shape} does not match CT size {expected_shape}.")

    contours: List[Tuple[int, List[float], int]] = []
    transform = reference_image.TransformContinuousIndexToPhysicalPoint
    depth = mask.shape[0]

    for z_index in range(depth):
        slice_mask = mask[z_index]
        if not np.any(slice_mask):
            continue
        paths = measure.find_contours(slice_mask.astype(np.float32), level=0.5)
        for path in paths:
            if path.shape[0] < 3:
                continue
            if not np.allclose(path[0], path[-1]):
                path = np.vstack([path, path[0]])

            contour_coords: List[float] = []
            for row, col in path:
                phys = transform((float(col), float(row), float(z_index)))
                contour_coords.extend([float(phys[0]), float(phys[1]), float(phys[2])])

            contours.append(
                (
                    z_index,
                    contour_coords,
                    len(path),
                )
            )

    if not contours:
        raise ValueError("Mask rasterisation did not produce any contours.")
    return contours


def _next_value(sequence: Sequence[Dataset], field_name: str, default_start: int = 1) -> int:
    values = [int(getattr(item, field_name, default_start - 1)) for item in sequence] if sequence else []
    return (max(values) + 1) if values else default_start


def _ensure_sequence(dataset: FileDataset, attr: str) -> DicomSequence:
    seq = getattr(dataset, attr, None)
    if seq is None:
        seq = DicomSequence([])
        setattr(dataset, attr, seq)
    return seq


def _extract_dataset(base: Union[chelp.RTStructData, FileDataset]) -> FileDataset:
    if isinstance(base, chelp.RTStructData):
        return base.dataset
    if isinstance(base, FileDataset):
        return base
    raise TypeError("Base RTSTRUCT must be a chelp.RTStructData or FileDataset.")


def embed_mask_as_structure(
    rtstruct: Union[chelp.RTStructData, FileDataset],
    *,
    mask: np.ndarray,
    reference_image: sitk.Image,
    slice_uid_map: SliceUIDMap,
    roi_name: str,
    color: Sequence[int] = (255, 0, 0),
    description: str | None = None,
    interpreted_type: str = "ISODOSE",
    generation_algorithm: str = "AUTOMATIC",
    new_uids: bool = False,
) -> FileDataset:
    """Return a new RTSTRUCT dataset with ``mask`` appended as a ROI."""

    if len(color) != 3:
        raise ValueError("ROI color must provide exactly three RGB components.")

    contours = mask_to_contours(mask, reference_image=reference_image)

    base_dataset = _extract_dataset(rtstruct)
    dataset = copy.deepcopy(base_dataset)
    structure_seq = getattr(dataset, "StructureSetROISequence", None)
    if not structure_seq:
        raise ValueError("RTSTRUCT dataset is missing StructureSetROISequence.")

    roi_numbers = [int(getattr(it, "ROINumber", 0)) for it in structure_seq]
    new_roi_number = max(roi_numbers) + 1 if roi_numbers else 1

    frame_uid = None
    if structure_seq:
        frame_uid = getattr(structure_seq[0], "ReferencedFrameOfReferenceUID", None)
    if frame_uid is None:
        ref_seq = getattr(dataset, "ReferencedFrameOfReferenceSequence", None)
        if ref_seq:
            frame_uid = getattr(ref_seq[0], "FrameOfReferenceUID", None)

    roi_item = Dataset()
    roi_item.ROINumber = new_roi_number
    if frame_uid is not None:
        roi_item.ReferencedFrameOfReferenceUID = frame_uid
    roi_item.ROIName = roi_name
    if description is not None:
        roi_item.ROIDescription = description
    roi_item.ROIGenerationAlgorithm = generation_algorithm
    structure_seq.append(roi_item)

    roi_contour_seq = _ensure_sequence(dataset, "ROIContourSequence")
    roi_contour = Dataset()
    roi_contour.ReferencedROINumber = new_roi_number
    roi_contour.ROIDisplayColor = [int(c) for c in color]
    roi_contour.ContourSequence = DicomSequence([])

    for contour_index, (z_index, contour_data, num_points) in enumerate(contours, start=1):
        slice_ref = slice_uid_map.get(z_index)
        if slice_ref is None:
            raise KeyError(f"No CT slice UID mapping for z-index {z_index}.")
        sop_instance_uid, sop_class_uid = slice_ref

        contour_ds = Dataset()
        contour_ds.ContourGeometricType = "CLOSED_PLANAR"
        contour_ds.NumberOfContourPoints = int(num_points)
        contour_ds.ContourData = [float(val) for val in contour_data]

        image_ds = Dataset()
        image_ds.ReferencedSOPClassUID = sop_class_uid
        image_ds.ReferencedSOPInstanceUID = sop_instance_uid
        contour_ds.ContourImageSequence = DicomSequence([image_ds])

        roi_contour.ContourSequence.append(contour_ds)

    roi_contour_seq.append(roi_contour)

    obs_seq = _ensure_sequence(dataset, "RTROIObservationsSequence")
    obs_item = Dataset()
    obs_item.ObservationNumber = _next_value(obs_seq, "ObservationNumber")
    obs_item.ReferencedROINumber = new_roi_number
    obs_item.RTROIInterpretedType = interpreted_type
    obs_item.ROIObservationLabel = roi_name
    obs_seq.append(obs_item)

    if new_uids:
        now = datetime.utcnow()
        dataset.SOPInstanceUID = generate_uid()
        if hasattr(dataset, "SeriesInstanceUID"):
            dataset.SeriesInstanceUID = generate_uid()
        dataset.InstanceCreationDate = now.strftime("%Y%m%d")
        dataset.InstanceCreationTime = now.strftime("%H%M%S")

    return dataset


def write_rtstruct(dataset: FileDataset, output_path: Path | str) -> Path:
    """Write the RTSTRUCT dataset to disk and return the saved path."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pydicom.dcmwrite(str(output_path), dataset, write_like_original=False)
    return output_path
