"""Utilities for accessing DICOM data used in TG-43 dose calculations."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import json
import pydicom
from pydicom.uid import generate_uid
import SimpleITK as sitk


@dataclass
class ChannelInfo:
    """Per-channel dwell metadata (positions, weights, relative offsets, strengths)."""

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
    use_cache: bool = True,
    nii_name: str = "ct_volume.nii.gz",
    json_name: str = "ct_metadata.json",
    return_numpy: bool = True,
) -> Tuple[sitk.Image, Optional[np.ndarray], Dict[str, Any]]:
    """Load a CT series (or cached NIfTI/JSON) and return the image, optional array, and metadata."""
    ct_dir = Path(ct_dir)
    if not ct_dir.exists():
        raise FileNotFoundError(f"CT directory not found: {ct_dir}")

    nii_path = ct_dir / nii_name
    json_path = ct_dir / json_name

    # --- Helper: build ct_metadata from sitk.Image ---
    def _meta_from_img(img: sitk.Image, source: str) -> Dict[str, Any]:
        return {
            "spacing": img.GetSpacing(),
            "origin": img.GetOrigin(),
            "direction": img.GetDirection(),
            "size": img.GetSize(),
            "source": source,
        }

    # --- Case 1: Cache hit (both files exist) ---
    if use_cache:
        if nii_path.exists() and json_path.exists():
            ct_image = sitk.ReadImage(str(nii_path))
            if orient_to:
                orienter = sitk.DICOMOrientImageFilter()
                orienter.SetDesiredCoordinateOrientation(orient_to)
                ct_image = orienter.Execute(ct_image)

            ct_array = sitk.GetArrayFromImage(ct_image).astype(np.int16) if return_numpy else None
            ct_metadata = _meta_from_img(ct_image, source="cache")
            # (Optional) read extra fields from JSON if you stored any
            try:
                with open(json_path, "r") as f:
                    extra = json.load(f)
                ct_metadata.update({k: v for k, v in extra.items() if k not in ct_metadata})
            except Exception:
                pass
            return ct_image, ct_array, ct_metadata

    # --- Case 2: Cache miss -> read DICOM, then write cache ---
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

    if use_cache: 
        # Save NIfTI cache (geometry is embedded in the header)
        sitk.WriteImage(ct_image, str(nii_path))

        # Save JSON sidecar (store whatever you need beyond geometry)
        ct_metadata = _meta_from_img(ct_image, source="dicom->cache")
        # You can add DICOM identifiers for traceability if desired:
        # e.g., ct_metadata["SeriesInstanceUID"] = chosen_uid
        with open(json_path, "w") as f:
            json.dump(ct_metadata, f, indent=2)

    ct_array = sitk.GetArrayFromImage(ct_image).astype(np.int16) if return_numpy else None
    return ct_image, ct_array, ct_metadata

def load_rtdose_volume(rtdose_path: Path | str) -> Tuple[sitk.Image, np.ndarray, Dict[str, Any]]:
    """Read an RTDOSE slice, apply ``DoseGridScaling``, and return the image, array, and geometry metadata."""
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
        "origin": dose_image.GetOrigin(),
        "direction": dose_image.GetDirection(),
        "size": dose_image.GetSize(),
    }
    return dose_image, dose_array, dose_metadata

def resample_to_reference(
    image: sitk.Image,
    reference: sitk.Image,
    default_value: float = 0.0,
) -> Tuple[sitk.Image, np.ndarray, Dict[str, Any]]:
    """Resample ``image`` onto ``reference`` using linear interpolation and copy its geometry."""
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
    """Resample ``image`` to the requested isotropic or per-axis spacing."""
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

def load_rtplan_by_channel(rtplan_path: Path | str, *, all_points: bool = False) -> Union[List[ChannelInfo], Tuple[List[ChannelInfo], Dict[str, Any]]]:
    """Parse an RTPLAN into per-channel dwell sequences with cumulative weights and strengths.

    When ``all_points`` is ``True`` the function also returns auxiliary metadata
    with dose reference points (e.g. Point A/B) and the RT plan description.
    """

    def _as_float(value: object) -> Optional[float]:
        try:
            if value in (None, ""):
                return None
            return float(value)
        except Exception:
            return None

    def _extract_reference_points(dataset: pydicom.dataset.Dataset) -> List[Dict[str, Any]]:
        points: List[Dict[str, Any]] = []
        for idx, ref in enumerate(getattr(dataset, "DoseReferenceSequence", [])):
            coords = getattr(ref, "DoseReferencePointCoordinates", None)
            if coords is None:
                continue
            coords_cm = np.asarray([_as_float(v) or 0.0 for v in coords], dtype=float) / 10.0
            number_val = _as_float(getattr(ref, "DoseReferenceNumber", None))
            roi_val = _as_float(getattr(ref, "ReferencedROINumber", None))
            point_entry: Dict[str, Any] = {
                "number": int(number_val) if number_val is not None else idx + 1,
                "description": str(getattr(ref, "DoseReferenceDescription", "") or "").strip(),
                "type": str(getattr(ref, "DoseReferenceType", "") or "").strip(),
                "positions_cm": coords_cm,
                "roi_number": int(roi_val) if roi_val is not None else None,
                "target_prescription_dose_Gy": _as_float(getattr(ref, "TargetPrescriptionDose", None)),
                "target_minimum_dose_Gy": _as_float(getattr(ref, "TargetMinimumDose", None)),
                "target_maximum_dose_Gy": _as_float(getattr(ref, "TargetMaximumDose", None)),
                "under_dose_volume_fraction": _as_float(getattr(ref, "TargetUnderDoseVolumeFraction", None)),
                "over_dose_volume_fraction": _as_float(getattr(ref, "TargetOverdoseVolumeFraction", None)),
            }
            uid = getattr(ref, "DoseReferenceUID", None)
            if uid:
                point_entry["uid"] = str(uid)
            points.append(point_entry)
        return points

    rtplan_path = Path(rtplan_path)
    if not rtplan_path.exists():
        raise FileNotFoundError(f"RTPLAN file not found: {rtplan_path}")

    ds = pydicom.dcmread(str(rtplan_path))
    if ds is None:
        raise ValueError("Unable to read RTPLAN dataset")

    plan_description = str(getattr(ds, "RTPlanDescription", "") or "").strip()

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

    if not all_points:
        return channels

    aux_points = {
        "dose_reference_points": _extract_reference_points(ds),
        "rtplan_description": plan_description,
    }
    return channels, aux_points


def save_rtplan_with_channels(
    template_path: Path | str,
    output_path: Path | str,
    channels: Sequence[ChannelInfo],
    *,
    update_uids: bool = True,
    anonymization: bool = True,
) -> None:
    """Copy an RTPLAN and overwrite dwell weights/times using ``channels`` (optionally anonymising/U-ID refreshing)."""

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

def extract_dwell_positions(ct_image: sitk.Image, channels: List[ChannelInfo], unique: bool = False) -> np.ndarray:
    """Map dwell positions (cm) into CT voxel indices for quick QA plotting."""

    positions = {"position":[], "channel":[]}
    for _, channel in enumerate(channels):
        indices: List[tuple[int, int, int]] = []
        for pos_cm in channel.positions_cm:
            if pos_cm is None:
                continue
            try:
                point_mm = tuple((pos_cm * 10.0).tolist())
                idx = ct_image.TransformPhysicalPointToIndex(point_mm)
                indices.append(idx)
            except RuntimeError:
                continue
        if unique:
            indices = list(set(indices))

        positions["position"].extend(indices)
        positions["channel"].extend([channel.channel_number] * len(indices))
    
    return positions
