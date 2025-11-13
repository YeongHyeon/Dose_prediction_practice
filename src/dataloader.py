import numpy as np

import tg43.dicom_helper as dhelp
import tg43.contour_helper as chelp

def load_case(paths: Dict[str, str]):

    # DICOM loading
    ## CT
    ct_image, ct_array, ct_metadata = dhelp.load_ct_volume(paths["ct"])

    ## RTDOSE
    dose_image, dose_array, dose_metadata = dhelp.load_rtdose_volume(paths['dose'])
    dose_image_r, dose_array_r, dose_metadata_r = dhelp.resample_to_reference(dose_image, ct_image)

    ## RTPLAN
    rt_channels = dhelp.load_rtplan_by_channel(paths["plan"])

    ## RTSTRUCT
    rtstruct = chelp.load_rtstruct(paths["struct"])


    # Pre-processing
    ## Rasterize the masks 
    structures = ["Bladder", "Rectum", "Sigmoid", "Bowel", "HR-CTV"]
    num_axial, num_coronal, num_sagittal = ct_array.shape
    num_class = len(structures)
    mask_array = np.zeros((num_axial, num_coronal, num_sagittal, num_class))
    for idx_structure, name_structure in enumerate(structures):
        mask_tmp = chelp.rasterise_structure(rtstruct, name_structure, ct_image)
        mask_array[..., idx_structure] = mask_tmp.astype(np.float32)

    ## Extract slices from volume by dwell positions
    ct_slices = []
    dose_slices = []
    mask_slices = []
    dwell_positions = dhelp.extract_dwell_positions(ct_image, rt_channels)
    for _, dwell_position in enumerate(dwell_positions):
        ct_slices.append(get_slice_by_dwell(ct_array, dwell_position, axis=0))
        dose_slices.append(get_slice_by_dwell(dose_array_r, dwell_position, axis=0))
        mask_slices.append(get_slice_by_dwell(mask_array, dwell_position, axis=0))


def get_slice_by_dwell(ct_array: np.ndarray, dwell_position: np.ndarray, axis: int = 0):
    """
    Extracts a 2D slice from a 3D CT array at the position of a dwell point along a specified axis.
    axis: 0 for axial, 1 for coronal, 2 for sagittal.
    """
    slice_index_rounded = int(np.clip(np.round(dwell_position[-1+axis]), 0, ct_array.shape[axis] - 1))
    slicer = [slice(None)] * 3
    slicer[axis] = slice_index_rounded
    return ct_array[tuple(slicer)]