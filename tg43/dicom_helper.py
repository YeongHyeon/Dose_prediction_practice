"""Utilities for accessing DICOM data used in TG-43 dose calculations."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pydicom
from pydicom.uid import generate_uid
import SimpleITK as sitk


@dataclass
class ChannelInfo:
    """Lightweight container for dwell control points in a single channel.

    Physical positions are stored in centimetres to align with the TG-43
    implementation, while time weights, strengths, and relative positions are
    captured as numpy arrays for vectorised downstream processing.
    """

    setup_number: int
    channel_number: int
    channel_id: str
    total_time_s: Optional[float]
    final_cumulative_weight: Optional[float]
    positions_cm: List[Optional[np.ndarray]]
    cumulative_weights: np.ndarray
    relative_positions: np.ndarray
    strengths_U: np.ndarray

def load_ct_volume(
    ct_dir: Path | str,
    series_uid: Optional[str] = None,
    orient_to: Optional[str] = None,
) -> Tuple[sitk.Image, np.ndarray, Dict[str, Any]]:
    """Load a DICOM CT series into a SimpleITK volume.

    The series is optionally reoriented, and the resulting SimpleITK image,
    NumPy array, and geometry metadata (spacing, origin, direction, size) are
    returned for downstream use. All metadata values remain in millimetres.

    Parameters
    ----------
    ct_dir : str or Path
        Directory containing the DICOM instances for a CT series.
    series_uid : str, optional
        SeriesInstanceUID to load when multiple series are present.
    orient_to : str, optional
        Target DICOM orientation code (for example ``"RAS"``) to enforce.

    Returns
    -------
    tuple[sitk.Image, numpy.ndarray, dict]
        The SimpleITK image, voxel array, and metadata for spacing, origin,
        direction, and size.

    Raises
    ------
    FileNotFoundError
        If ``ct_dir`` does not exist.
    RuntimeError
        If the directory does not contain a readable CT series.
    ValueError
        If ``series_uid`` is provided but not found in the directory.
    """
    ct_dir = Path(ct_dir)
    if not ct_dir.exists():
        raise FileNotFoundError(f"CT directory not found: {ct_dir}")

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(ct_dir))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in: {ct_dir}")

    chosen_uid = series_uid if series_uid is not None else series_ids[0]
    if chosen_uid not in series_ids:
        raise ValueError(
            f"Requested SeriesInstanceUID {chosen_uid} not among available {series_ids}"
        )

    file_names = reader.GetGDCMSeriesFileNames(str(ct_dir), chosen_uid)
    if not file_names:
        raise RuntimeError(
            f"No readable DICOM slices for series {chosen_uid} in {ct_dir}"
        )

    reader.SetFileNames(file_names)
    ct_image = reader.Execute()

    if orient_to:
        orienter = sitk.DICOMOrientImageFilter()
        orienter.SetDesiredCoordinateOrientation(orient_to)
        ct_image = orienter.Execute(ct_image)

    ct_array = sitk.GetArrayFromImage(ct_image).astype(np.int16)
    ct_metadata = {
        "spacing": ct_image.GetSpacing(),
        "origin": ct_image.GetOrigin(),
        "direction": ct_image.GetDirection(),
        "size": ct_image.GetSize(),
    }
    return ct_image, ct_array, ct_metadata

def load_rtdose_volume(rtdose_path: Path | str) -> Tuple[sitk.Image, np.ndarray, Dict[str, Any]]:
    """Load an RTDOSE dataset into a SimpleITK image.

    Pixel data are scaled using ``DoseGridScaling`` and wrapped in a
    SimpleITK image whose spacing/origin/direction mirror the RTDOSE tags. The
    companion array is returned in gray for direct analysis.

    Parameters
    ----------
    rtdose_path : str or Path
        Location of the RTDOSE DICOM file on disk.

    Returns
    -------
    tuple[sitk.Image, numpy.ndarray, dict]
        The SimpleITK image, voxel array in gray, and geometry metadata.

    Raises
    ------
    FileNotFoundError
        If ``rtdose_path`` does not exist.
    """
    rtdose_path = Path(rtdose_path)
    if not rtdose_path.exists():
        raise FileNotFoundError(f"RTDOSE file not found: {rtdose_path}")

    dcm = pydicom.dcmread(str(rtdose_path))
    dose = dcm.pixel_array.astype(np.float32)
    dose *= float(getattr(dcm, 'DoseGridScaling', 1.0))

    dy, dx = (float(v) for v in dcm.PixelSpacing)
    orientation = np.array([float(v) for v in dcm.ImageOrientationPatient])
    row = orientation[0:3]
    col = orientation[3:6]
    slice_dir = np.cross(row, col)
    ipp0 = np.array([float(v) for v in dcm.ImagePositionPatient])

    gfv = [float(v) for v in getattr(dcm, 'GridFrameOffsetVector', [])]
    if len(gfv) > 1:
        dz = float(np.mean(np.diff(gfv)))
    elif len(gfv) == 1:
        dz = float(getattr(dcm, 'SliceThickness', gfv[0] or 1.0))
    else:
        dz = float(getattr(dcm, 'SliceThickness', 1.0))

    dose_image = sitk.GetImageFromArray(dose)
    dose_image.SetSpacing((dx, dy, abs(dz)))
    dose_image.SetOrigin(tuple(ipp0))
    direction = np.vstack([row, col, slice_dir]).T.flatten()
    dose_image.SetDirection(tuple(direction))

    dose_array = sitk.GetArrayFromImage(dose_image).astype(np.float32)
    dose_metadata = {
        "spacing": dose_image.GetSpacing(),
        "origin": np.array(dose_image.GetOrigin()),
        "direction": dose_image.GetDirection(),
        "offsets": np.array(gfv),
        "size": dose_image.GetSize(),
    }
    return dose_image, dose_array, dose_metadata

def resample_to_reference(
    image: sitk.Image,
    reference: sitk.Image,
    default_value: float = 0.0,
) -> Tuple[sitk.Image, np.ndarray, Dict[str, Any]]:
    """Resample a SimpleITK image onto a reference grid.

    Uses linear interpolation and copies the reference geometry so the returned
    image and NumPy array align with downstream CT volumes. Voxels outside the
    source image footprint are filled with ``default_value``.

    Parameters
    ----------
    image : SimpleITK.Image
        Image that needs to be resampled.
    reference : SimpleITK.Image
        Reference geometry that defines the desired spacing and origin.
    default_value : float, optional
        Fill value for voxels that map outside the source extent.

    Returns
    -------
    tuple[sitk.Image, numpy.ndarray, dict]
        The resampled image, voxel array, and accompanying metadata.
    """
    resampled_image = sitk.Resample(
        image,
        reference,
        sitk.Transform(),
        sitk.sitkLinear,
        default_value,
        image.GetPixelID(),
    )

    resampled_array = sitk.GetArrayFromImage(resampled_image).astype(np.float32)
    resampled_metadata = {
        "spacing": resampled_image.GetSpacing(),
        "origin": resampled_image.GetOrigin(),
        "direction": resampled_image.GetDirection(),
        "size": resampled_image.GetSize(),
    }
    return resampled_image, resampled_array, resampled_metadata

def resample_to_spacing(
    image: sitk.Image,
    spacing: Sequence[float] | float,
    default_value: float = 0.0,
    interpolator=sitk.sitkLinear,
) -> Tuple[sitk.Image, np.ndarray, Dict[str, Any]]:
    """Resample a SimpleITK image onto a grid with user-defined spacing.

    Parameters
    ----------
    image : SimpleITK.Image
        Image that needs to be resampled.
    spacing : float or Sequence[float]
        Desired voxel spacing in millimetres. When a single float is provided,
        it is applied uniformly across all dimensions.
    default_value : float, optional
        Fill value for voxels that map outside the source extent.
    interpolator : SimpleITK interpolator enum, optional
        Interpolator passed through to ``sitk.Resample``.

    Returns
    -------
    tuple[sitk.Image, numpy.ndarray, dict]
        The resampled image, voxel array, and accompanying metadata.

    Raises
    ------
    ValueError
        If the spacing values are non-positive or do not match the image
        dimensionality.
    """
    dimension = image.GetDimension()

    if np.isscalar(spacing):
        target_spacing = (float(spacing),) * dimension
    else:
        try:
            target_spacing = tuple(float(s) for s in spacing)
        except TypeError as exc:  # pragma: no cover - defensive
            raise TypeError("Spacing must be a float or sequence of floats.") from exc

    if len(target_spacing) != dimension:
        raise ValueError(
            f"Received spacing of length {len(target_spacing)} for {dimension}D image."
        )
    if any(s <= 0 for s in target_spacing):
        raise ValueError("Spacing values must be strictly positive.")

    original_spacing = np.array(image.GetSpacing(), dtype=np.float64)
    original_size = np.array(image.GetSize(), dtype=np.int64)
    target_spacing_arr = np.array(target_spacing, dtype=np.float64)

    physical_extent = original_spacing * np.maximum(original_size - 1, 0)
    target_size = np.ceil(physical_extent / target_spacing_arr).astype(np.int64) + 1
    target_size = tuple(int(max(1, n)) for n in target_size)

    resampled_image = sitk.Resample(
        image,
        target_size,
        sitk.Transform(),
        interpolator,
        image.GetOrigin(),
        target_spacing,
        image.GetDirection(),
        default_value,
        image.GetPixelID(),
    )

    resampled_array = sitk.GetArrayFromImage(resampled_image).astype(np.float32)
    resampled_metadata = {
        "spacing": resampled_image.GetSpacing(),
        "origin": resampled_image.GetOrigin(),
        "direction": resampled_image.GetDirection(),
        "size": resampled_image.GetSize(),
    }
    return resampled_image, resampled_array, resampled_metadata

def load_rtplan_by_channel(rtplan_path: Path | str) -> List[ChannelInfo]:
    """Extract dwell control points from an RTPLAN by channel.

    Parses brachytherapy application setups, folds in overrides from the
    FractionGroupSequence, and yields ``ChannelInfo`` records containing dwell
    positions, cumulative weights, and per-control-point strengths. Channels
    are returned in the order they appear in the RTPLAN dataset.

    Parameters
    ----------
    rtplan_path : str or Path
        Path to the RTPLAN DICOM dataset.

    Returns
    -------
    list[ChannelInfo]
        Channel-wise dwell metadata including times, positions, and weights.

    Raises
    ------
    FileNotFoundError
        If the RTPLAN file is missing.
    ValueError
        If the dataset cannot be read as DICOM.
    """

    def _as_float(value: object) -> Optional[float]:
        try:
            if value in (None, ""):
                return None
            return float(value)
        except Exception:
            return None

    rtplan_path = Path(rtplan_path)
    if not rtplan_path.exists():
        raise FileNotFoundError(f"RTPLAN file not found: {rtplan_path}")

    ds = pydicom.dcmread(str(rtplan_path))
    if ds is None:
        raise ValueError("Unable to read RTPLAN dataset")

    if hasattr(ds, "BrachyApplicationSetupSequence"):
        setups = ds.BrachyApplicationSetupSequence
    elif hasattr(ds, "ApplicationSetupSequence"):
        setups = ds.ApplicationSetupSequence
    else:
        return []

    # Collect overrides from FractionGroupSequence (per-channel total time / strengths)
    overrides: dict[tuple[int, int], dict[str, float]] = {}
    if hasattr(ds, "FractionGroupSequence"):
        for fg in ds.FractionGroupSequence:
            for ref_setup in getattr(fg, "ReferencedBrachyApplicationSetupSequence", []):
                setup_no = _as_float(getattr(ref_setup, "ReferencedApplicationSetupNumber", None))
                if setup_no is None:
                    continue
                for ref_channel in getattr(ref_setup, "ChannelSequence", []):
                    channel_no = _as_float(getattr(ref_channel, "ReferencedChannelNumber", None))
                    if channel_no is None:
                        continue
                    key = (int(setup_no), int(channel_no))
                    overrides[key] = {
                        "channel_total_time": _as_float(getattr(ref_channel, "ChannelTotalTime", None)) or 0.0,
                        "nominal_strength": _as_float(getattr(ref_channel, "NominalSourceStrength", None)) or 0.0,
                        "air_kerma_strength": _as_float(getattr(ref_channel, "AirKermaStrength", None)) or 0.0,
                        "reference_air_kerma_rate": _as_float(getattr(ref_channel, "ReferenceAirKermaRate", None)) or 0.0,
                    }

    plan_fallback = {
        "nominal_strength": 0.0,
        "air_kerma_strength": 0.0,
        "reference_air_kerma_rate": 0.0,
    }
    if hasattr(ds, "SourceSequence") and ds.SourceSequence:
        src = ds.SourceSequence[0]
        plan_fallback = {
            "nominal_strength": _as_float(getattr(src, "NominalSourceStrength", None)) or 0.0,
            "air_kerma_strength": _as_float(getattr(src, "AirKermaStrength", None)) or 0.0,
            "reference_air_kerma_rate": _as_float(getattr(src, "ReferenceAirKermaRate", None)) or 0.0,
        }

    channels: List[ChannelInfo] = []
    for setup_idx, setup in enumerate(setups):
        setup_number = int(getattr(setup, "ApplicationSetupNumber", setup_idx + 1))
        channel_seq = getattr(setup, "ChannelSequence", [])
        for channel_idx, channel in enumerate(channel_seq):
            channel_number = int(getattr(channel, "ChannelNumber", channel_idx + 1))
            channel_id = getattr(channel, "ChannelIdentifier", f"Channel_{channel_number}")
            key = (setup_number, channel_number)
            override = overrides.get(key, {})

            total_time = override.get("channel_total_time")
            if not total_time:
                total_time = _as_float(getattr(channel, "ChannelTotalTime", None))

            final_weight = _as_float(getattr(channel, "FinalCumulativeTimeWeight", None))

            def _pick_strength(*values: Optional[float]) -> float:
                for val in values:
                    if val is not None and val > 0:
                        return float(val)
                return 0.0

            base_strength = _pick_strength(
                override.get("air_kerma_strength"),
                _as_float(getattr(channel, "AirKermaStrength", None)),
                plan_fallback["air_kerma_strength"],
                override.get("nominal_strength"),
                _as_float(getattr(channel, "NominalSourceStrength", None)),
                plan_fallback["nominal_strength"],
                override.get("reference_air_kerma_rate"),
                _as_float(getattr(channel, "ReferenceAirKermaRate", None)),
                plan_fallback["reference_air_kerma_rate"],
            )

            positions: List[Optional[np.ndarray]] = []
            weights: List[float] = []
            rel_pos: List[float] = []
            strengths: List[float] = []

            for cp in getattr(channel, "BrachyControlPointSequence", []):
                pos = getattr(cp, "ControlPoint3DPosition", None)
                positions.append(np.asarray(pos, dtype=float) / 10.0 if pos is not None else None)
                weights.append(_as_float(getattr(cp, "CumulativeTimeWeight", None)) or 0.0)
                rel_pos.append(_as_float(getattr(cp, "ControlPointRelativePosition", None)) or 0.0)

                cp_strength = _pick_strength(
                    _as_float(getattr(cp, "AirKermaStrength", None)),
                    _as_float(getattr(cp, "NominalSourceStrength", None)),
                    _as_float(getattr(cp, "ReferenceAirKermaRate", None)),
                    base_strength,
                )
                strengths.append(cp_strength)

            if not weights:
                continue

            channels.append(
                ChannelInfo(
                    setup_number=setup_number,
                    channel_number=channel_number,
                    channel_id=str(channel_id),
                    total_time_s=total_time,
                    final_cumulative_weight=final_weight,
                    positions_cm=positions,
                    cumulative_weights=np.asarray(weights, dtype=float),
                    relative_positions=np.asarray(rel_pos, dtype=float),
                    strengths_U=np.asarray(strengths, dtype=float),
                )
            )

    return channels


def save_rtplan_with_channels(
    template_path: Path | str,
    output_path: Path | str,
    channels: Sequence[ChannelInfo],
    *,
    update_uids: bool = True,
    anonymization: bool = True,
) -> None:
    """Write a new RTPLAN DICOM with dwell times updated from ``channels``.

    Parameters
    ----------
    template_path : str or Path
        Source RTPLAN file used as the structural template.
    output_path : str or Path
        Destination filename for the updated RTPLAN.
    channels : Sequence[ChannelInfo]
        Channel descriptors containing cumulative dwell times (seconds).
    update_uids : bool, optional
        When ``True`` new SOP/Series instance UIDs are generated to avoid clashes.

    Raises
    ------
    ValueError
        If the RTPLAN structure does not match the supplied channel information.
    """

    template_path = Path(template_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = pydicom.dcmread(str(template_path))

    channel_map: Dict[Tuple[int, int], ChannelInfo] = {
        (int(info.setup_number), int(info.channel_number)): info for info in channels
    }

    consumed: set[Tuple[int, int]] = set()

    setups = []
    if hasattr(ds, "BrachyApplicationSetupSequence"):
        setups = ds.BrachyApplicationSetupSequence
    elif hasattr(ds, "ApplicationSetupSequence"):
        setups = ds.ApplicationSetupSequence

    for setup in setups:
        setup_number = int(getattr(setup, "ApplicationSetupNumber", 0))
        channel_seq = getattr(setup, "ChannelSequence", [])
        for channel in channel_seq:
            channel_number = int(getattr(channel, "ChannelNumber", 0))
            key = (setup_number, channel_number)
            info = channel_map.get(key)
            if info is None:
                raise ValueError(f"Missing ChannelInfo for setup {setup_number}, channel {channel_number}")
            consumed.add(key)

            weights = np.asarray(info.cumulative_weights, dtype=float)
            total_time = float(weights[-1]) if weights.size else 0.0
            if total_time > 0:
                normalized = weights / total_time
                normalized[-1] = 1.0
            else:
                normalized = weights

            if total_time > 0:
                channel.ChannelTotalTime = float(total_time)
            elif hasattr(channel, "ChannelTotalTime"):
                channel.ChannelTotalTime = float(total_time)

            if normalized.size:
                channel.FinalCumulativeTimeWeight = float(normalized[-1])

            cp_sequence = getattr(channel, "BrachyControlPointSequence", [])
            if len(cp_sequence) != normalized.size:
                raise ValueError(
                    f"Cumulative weights mismatch for setup {setup_number}, channel {channel_number}: "
                    f"{normalized.size} weights vs {len(cp_sequence)} control points"
                )
            for cp, weight in zip(cp_sequence, normalized):
                cp.CumulativeTimeWeight = float(weight)

    if consumed != set(channel_map):
        missing = set(channel_map) - consumed
        raise ValueError(f"Channel descriptors not written: {sorted(missing)}")

    if hasattr(ds, "FractionGroupSequence"):
        for fg in ds.FractionGroupSequence:
            for ref_setup in getattr(fg, "ReferencedBrachyApplicationSetupSequence", []):
                setup_no = int(getattr(ref_setup, "ReferencedApplicationSetupNumber", 0))
                for ref_channel in getattr(ref_setup, "ChannelSequence", []):
                    channel_no = int(getattr(ref_channel, "ReferencedChannelNumber", 0))
                    info = channel_map.get((setup_no, channel_no))
                    if info is None:
                        continue
                    weights = np.asarray(info.cumulative_weights, dtype=float)
                    total_time = float(weights[-1]) if weights.size else 0.0
                    if total_time > 0:
                        ref_channel.ChannelTotalTime = float(total_time)
                        ref_channel.FinalCumulativeTimeWeight = 1.0
                    else:
                        ref_channel.ChannelTotalTime = float(total_time)
                        ref_channel.FinalCumulativeTimeWeight = 0.0

    if update_uids:
        ds.SOPInstanceUID = generate_uid()
        if hasattr(ds, "StudyInstanceUID"):
            ds.StudyInstanceUID = generate_uid()
        if hasattr(ds, "SeriesInstanceUID"):
            ds.SeriesInstanceUID = generate_uid()
    if anonymization:
        if hasattr(ds, "PatientName"):
            ds.PatientName = "Anonymized"
        if hasattr(ds, "PatientID"):
            ds.PatientID = "Anonymized"
        if hasattr(ds, "PatientBirthDate"):
            ds.PatientBirthDate = "Anonymized"
        if hasattr(ds, "PatientSex"):
            ds.PatientSex = "Anonymized"
        if hasattr(ds, "AccessionNumber"):
            ds.AccessionNumber = "Anonymized"
        if hasattr(ds, "OperatorsName"):
            ds.OperatorsName = "Anonymized"
        if hasattr(ds, "ReferringPhysicianName"):
            ds.ReferringPhysicianName = "Anonymized"
        if hasattr(ds, "StudyID"):
            ds.StudyID = "Anonymized"

    ds.save_as(str(output_path))

def extract_dwell_positions(ct_image: sitk.Image, channels: List[ChannelInfo]) -> np.ndarray:
    """Convert dwell positions to CT voxel indices for QA plots.

    Parameters
    ----------
    ct_image : SimpleITK.Image
        CT image that defines the spatial index transform.
    channels : list[ChannelInfo]
        Dwell records whose physical positions are expressed in centimetres.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N, 3)`` containing continuous ``(z, y, x)`` indices
        for the dwells that map inside the CT volume. Empty arrays are returned
        when no positions fall inside the image bounds.
    """
 
    indices: List[tuple[float, float, float]] = []
    for channel in channels:
        for pos_cm in channel.positions_cm:
            if pos_cm is None:
                continue
            try:
                point_mm = tuple((pos_cm * 10.0).tolist())
                # Use the continuous transform to retain sub-voxel precision and
                # avoid the rounding artefacts of TransformPhysicalPointToIndex.
                idx = ct_image.TransformPhysicalPointToContinuousIndex(point_mm)
                indices.append(tuple(float(v) for v in idx))
            except RuntimeError:
                continue
    indices = list(set(indices))  # Remove duplicates
    return np.asarray(indices, dtype=float)
