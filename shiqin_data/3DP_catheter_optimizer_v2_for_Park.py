# %%
import os
import sys
import math
import json
import platform
import copy
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.draw import polygon2mask
import cvxpy as cp

# ==========================================================
""" Prescription and OAR Limits """
# ==========================================================
bt_dose_per_fx = 600  # cGy per fraction
bt_num_of_fx = 5     # total number of fractions

bladder_ebrt_dose = 4500  # cGy from EBRT
rectum_ebrt_dose  = 4500  # cGy from EBRT
sigmoid_ebrt_dose = 4500  # cGy from EBRT
bowel_ebrt_dose = 4500    # cGy from EBRT
rectovaginal_ebrt_dose = 4500    # cGy from EBRT
pt_surface_ebrt_dose = 4500  # cGy from EBRT
pt_5mm_ebrt_dose = 4500      # cGy from EBRT

bladder_limit = 8000   # cGy total limit EQD2
rectum_limit  = 6500   # cGy total limit EQD2
sigmoid_limit = 7000   # cGy total limit EQD2
bowel_limit   = 7000   # cGy total limit EQD2
rectovaginal_limit = 6500   # cGy total limit EQD2
pt_surface_limit_bt = bt_dose_per_fx * 1.4  # cGy 140% of prescription
pt_5mm_limit = 8500   # cGy total limit EQD2

def brachy_oar_limit_per_fraction(eqd2_limit_cGy, ebrt_dose_cGy, n_brachy_fractions, alpha_beta=3):  
    # Calculate the per-fraction physical dose limit (in cGy) for OAR in HDR brachytherapy.

    # Convert everything to Gy for calculation clarity
    eqd2_limit = eqd2_limit_cGy / 100.0
    ebrt_dose = ebrt_dose_cGy / 100.0

    # EQD2 for EBRT (25 fx)
    d_ebrt = ebrt_dose / 25.0
    eqd2_ebrt = 25.0 * d_ebrt * (d_ebrt + alpha_beta) / (2 + alpha_beta)

    # Remaining EQD2 budget available for brachy
    remaining_eqd2 = eqd2_limit - eqd2_ebrt

    # For HDR, find physical dose per fraction that gives total EQD2 = remaining_eqd2
    # EQD2_brachy = n * d * (1 + d/ab)
    # => d = [-ab + sqrt(ab^2 + 4 * (remaining_eqd2/n) * ab)] / 2
    term = alpha_beta**2 + 4 * (remaining_eqd2 / n_brachy_fractions) * (2 + alpha_beta)
    d_per_fraction = (-alpha_beta + math.sqrt(term)) / 2

    return d_per_fraction * 100.0  # return in cGy

bladder_limit_bt = brachy_oar_limit_per_fraction(bladder_limit, bladder_ebrt_dose, bt_num_of_fx)   # cGy per fx
rectum_limit_bt  = brachy_oar_limit_per_fraction(rectum_limit, rectum_ebrt_dose, bt_num_of_fx)   # cGy per fx
sigmoid_limit_bt = brachy_oar_limit_per_fraction(sigmoid_limit, sigmoid_ebrt_dose, bt_num_of_fx)  # cGy per fx
bowel_limit_bt   = brachy_oar_limit_per_fraction(bowel_limit, bowel_ebrt_dose, bt_num_of_fx)     # cGy per fx
rectovaginal_limit_bt = brachy_oar_limit_per_fraction(rectovaginal_limit, rectovaginal_ebrt_dose, bt_num_of_fx)   # cGy per fx   # cGy per fx
pt_5mm_limit_bt = brachy_oar_limit_per_fraction(pt_5mm_limit, pt_5mm_ebrt_dose, bt_num_of_fx)   # cGy per fx

# oar_limits = {  # per-fraction, EQD2 2cc limit
#     "bladder": bladder_limit_bt,  
#     "rectum":  rectum_limit_bt,
#     "sigmoid": sigmoid_limit_bt,
#     "bowel": bowel_limit_bt
# }

oars = {}
ctvs = {}
oar_names = ["bladder", "rectum", "sigmoid", "bowel"]
ctv_names = ["hr-ctv", "ir-ctv"]

oars["bladder"] = {"name": "bladder", "goal": bladder_limit_bt}
oars["rectum"] = {"name": "rectum", "goal": rectum_limit_bt}
oars["sigmoid"] = {"name": "sigmoid", "goal": sigmoid_limit_bt}
oars["bowel"] = {"name": "bowel", "goal": bowel_limit_bt}
ctvs["hr-ctv"] = {"name": "hr-ctv", "goal": bt_dose_per_fx}

# ==========================================================
""" Default settings """
# ==========================================================
vox_size = np.array([0.25, 0.25, 0.25])  # isotropic 0.25 cm
step_cm = 0.5 # step size
max_len_cm = 6.0 # max needle length

# ==========================================================
""" File paths """
# ==========================================================
if platform.system() == "Windows":
    base_path = r"C:\Users\ssu3\OneDrive - Inside MD Anderson\Projects\3D printing\3DP01"
    ovoid_config_path = r"C:\Users\ssu3\OneDrive - Inside MD Anderson\VSCode\Geneva_ovoid_dimension.json"
else:
    base_path = "/Users/ssu3/Library/CloudStorage/OneDrive-InsideMDAnderson/Projects/3D printing/3DP01"
    # update this path as needed on Mac if you mirror the JSON there
    ovoid_config_path = "/Users/ssu3/OneDrive - Inside MD Anderson/VSCode/Geneva_ovoid_dimension.json"

rp_path = os.path.join(base_path, "ima_empty_uid_1.3.6.1.4.1.2452.6.1862592041.1296722971.2117956285.1860453842.dcm")
rs_path = os.path.join(base_path, "RS_3DP01_Fx1.dcm")

rp = pydicom.dcmread(rp_path)
rs = pydicom.dcmread(rs_path)

# ==========================================================
""" Read catheters in CT coordinates """
# ==========================================================
if hasattr(rp, "ApplicationSetupSequence"):
    app = rp.ApplicationSetupSequence[0]
    channels = []
    if hasattr(app, "ChannelSequence"):
        for ch in app.ChannelSequence:
            ch_num = int(ch.ChannelNumber)
            ch_total_time = float(ch.ChannelTotalTime)
            ch_cum_time_weight = float(ch.FinalCumulativeTimeWeight)
            cps = ch.BrachyControlPointSequence

            # Extract cumulative weights and 3D positions
            cumw = np.array([float(cp.CumulativeTimeWeight) for cp in cps])
            pos  = np.array([cp.ControlPoint3DPosition for cp in cps])

            dwell_pos, dwell_time = [], []
            # Step through every 2 points (each dwell pair)
            for i in range(0, len(cps) - 1, 2):
                p = pos[i + 1]  # position of the second (active) control point
                dt = ch_total_time * (cumw[i + 1] - cumw[i]) / ch_cum_time_weight
                dwell_pos.append(p)
                dwell_time.append(dt)

            dwell_pos = np.array(dwell_pos, float) / 10  # convert to cm
            dwell_time = np.array(dwell_time, float)    # in seconds

            channels.append({
                "channel_number": ch_num,
                "dwell_position": dwell_pos,
                "dwell_time": dwell_time,
            })
else:
    sys.exit()
print(f"Found {len(channels)} channels in plan.")

Sk = rp.SourceSequence[0].ReferenceAirKermaRate
Lambda = 1.108  # dose rate constant cGy/h/U for Ir-192

# get dwell positions and times for ovoids + tandem only
dwell_pos_to = np.concatenate([ch["dwell_position"] for ch in channels[:3]])
dwell_time_to = np.concatenate([ch["dwell_time"] for ch in channels[:3]])

# create loopkup table for dwell gloabal index to dwell channel index
dwell_lookup = {}          # maps (channel_number, local_idx) → stacked_idx
idx = 0
for ch in channels:
    ch_num = ch["channel_number"]
    n_dwells = len(ch["dwell_position"])

    for local_idx in range(n_dwells):
        # store mapping
        dwell_lookup[(ch_num, local_idx)] = idx
        idx += 1

# ==========================================================
""" Read ROIs in CT coordinates """
# ==========================================================
# roi_keep = ["hr_ctv", "ir_ctv", "bladder", "rectum", "sigmoid", "bowel"]
# structures = {}
for roi, contour in zip(rs.StructureSetROISequence, rs.ROIContourSequence):
    name = roi.ROIName.strip().lower()
    if hasattr(contour, "ContourSequence"):
        all_pts = []
        for c in contour.ContourSequence:
            pts = np.array(c.ContourData, dtype=float).reshape(-1, 3) / 10  # convert to cm
            all_pts.append(pts)
        if name in oar_names:
            oars[name]["contour"] = all_pts
        elif name in ctv_names:
            ctvs[name]["contour"] = all_pts
        # structures[name] = all_pts

# # keep only the ones we care about
# structures = {
#     k: v for k, v in structures.items()
#     if any(x in k.lower() for x in roi_keep)
# }

# ==========================================================
""" Read point As (from RP) in CT coordinates """
# ==========================================================
pointsA = {}
if hasattr(rp, "DoseReferenceSequence"):
    for dr in rp.DoseReferenceSequence:
        desc = getattr(dr, "DoseReferenceDescription", "").upper()
        coords = getattr(dr, "DoseReferencePointCoordinates", None)
        if coords is not None and desc:
            pt = np.array(coords, dtype=float).reshape(1, 3)[0] / 10  # convert to cm
            if "A-R" in desc or "ART" in desc or "RIGHT" in desc:
                pointsA["ART"] = pt
            elif "A-L" in desc or "ALT" in desc or "LEFT" in desc:
                pointsA["ALT"] = pt
            elif "ICRU" in desc or "RECT" in desc or "VAGINA" in desc:
                pointRecto = pt
print("Found point As and ICRU recto-vaginal point.")

# ==================================================================
""" Create solid masks for structures in CT coordinates """
# ==================================================================
def build_solid_mask(contour, origin, dims, vox_size):
    # Convert DICOM ROI contour list into a filled 3D solid mask.
    mask = np.zeros(dims, dtype=bool)
    for poly in contour:
        # create mask for each polygon
        vox_idx = np.round((poly - origin) / vox_size).astype(int)  # convert cm → voxel index
        xy_idx = vox_idx[:, [0, 1]]  # x-y plane
        z_idx = np.unique(vox_idx[:,2]).item()

        if xy_idx.shape[0] < 3:
            for (x, y) in xy_idx:
                mask[x, y, z_idx] = True  # a valid polygon must have >=3 points
        else:
            mask[:, :, z_idx] |= polygon2mask([dims[0], dims[1]], xy_idx)
    
    # fill any interior holes in 3D
    mask = ndi.binary_fill_holes(mask)
    return mask

all_pts = []
all_oar_pts = []
for ctv in ctvs.values():
    for pts in ctv["contour"]:
        all_pts.append(np.asarray(pts))
for oar in oars.values():
    for pts in oar["contour"]:
        all_pts.append(np.asarray(pts))
        all_oar_pts.append(np.asarray(pts))
all_pts = np.vstack(all_pts)
all_oar_pts = np.vstack(all_oar_pts)

origin = all_pts.min(axis=0) - 0.5  # leave 0.5 cm margin
maxs = all_pts.max(axis=0) + 0.5
dims = np.ceil((maxs - origin) / vox_size).astype(int)
dims = np.maximum(dims, 1)
origin = origin + vox_size / 2  # center of first voxel

hrctv_mask = build_solid_mask(ctvs["hr-ctv"]["contour"], origin, dims, vox_size)
ctvs["hr-ctv"]["mask"] = hrctv_mask
# Only compute mask within 2 cm of HR-CTV for OAR dose optimization
dist_map = ndi.distance_transform_edt(~hrctv_mask, sampling=vox_size)

for oar in oars.values():
    name  = oar["name"]
    mask = build_solid_mask(oar["contour"], origin, dims, vox_size)
    mask_near_ctv = np.logical_and(mask, dist_map <= 2.0)
    oars[name]["mask"] = mask_near_ctv

# ==========================================================
""" Build ECS view
    Define a local frame:
    +x_axis = A_lt
    +y_axis = rectum
    +z_axis = tip of tandem
    Then translate so that intersection of tandem with A-plane goes to [0,0,20] mm. """
# ==========================================================
if len(channels) >= 3 and "ART" in pointsA and "ALT" in pointsA:
    tandem = channels[2]['dwell_position']  # assume catheter 3 = tandem
    L1, L2 = tandem[0], tandem[-1]

    # Find vectors for new axes
    z_axis = (L1 - L2) / np.linalg.norm(L1 - L2)  # +z_axis points from last dwell toward first dwell (superior)
    x_axis = (pointsA["ALT"] - pointsA["ART"])
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Find intersection of tandem line with the A-point plane
    # Plane normal is z_axis, plane goes through ART
    d_plane = np.dot(z_axis, pointsA["ART"])
    d_norm = np.dot(z_axis, (L2 - L1))
    if abs(d_norm) > 1e-6:
        t_plane = (d_plane - np.dot(z_axis, L1)) / d_norm
    else:
        t_plane = 0.0
    intersect = L1 + t_plane * (L2 - L1)

    # Rotation matrix: rows are new axes so that new_coord = R * old + T
    R = np.vstack([x_axis, y_axis, z_axis])

    # Translation: send that intersection to [0,0,2], where plane of point As and tandem intersects
    intersect_ecs = np.array([0.0, 0.0, 2.0])
    T = intersect_ecs - R @ intersect

    # Define transform (DICOM → Applicator)
    def transform_points(pts_dicom):
        pts_dicom = np.array(pts_dicom, dtype=float)
        return (R @ pts_dicom.T).T + T
    
    # Define inverse transform (Applicator → DICOM)
    def inverse_transform_points(pts_ecs):
        pts_ecs = np.array(pts_ecs, dtype=float)
        return (R.T @ (pts_ecs - T).T).T

    # Apply transform
    channels_ecs = [transform_points(ch["dwell_position"]) for ch in channels]
    pointsA_ecs = {
        k: transform_points(v[np.newaxis, :])[0]
        for k, v in pointsA.items()
    }
else:
    print("⚠️ Missing catheters or A-points. Coordinate transformation stoped.")
    sys.exit()

# dwell_pts_to = np.vstack(channels_tr[:3]) # transformed dwell points of ovoids + tandem
# dwell_time_to = np.concatenate([ch["dwell_times"] for ch in channels[:3]])

# ==========================================================
""" Compute nominal tandem angle
    Angle = angle between:
    v_tandem = (tandem last → first)
    v_ovoid  = (ovoid third → first)
    
    Do for right and left ovoid, average.
    Nominal_tandem_angle = mean_angle - 60
    Snap nominal_tandem_angle to {0, 15, 30, 45} """ 
# ==========================================================
nominal_tandem_angle = None
if len(channels_ecs) >= 3:
    ovoid_r_ecs = channels_ecs[0]  # Channel 1
    ovoid_l_ecs = channels_ecs[1]  # Channel 2
    tandem_ecs = channels_ecs[2]  # Channel 3

    if ovoid_r_ecs.shape[0] >= 3 and ovoid_l_ecs.shape[0] >= 3 and tandem_ecs.shape[0] >= 2:
        # Define direction vectors
        v_tandem = tandem_ecs[0] - tandem_ecs[-1]       # last → first (superior direction)
        v_tandem /= np.linalg.norm(v_tandem)

        v_ovoid_r = ovoid_r_ecs[0] - ovoid_r_ecs[2]     # third → first
        v_ovoid_r /= np.linalg.norm(v_ovoid_r)

        v_ovoid_l = ovoid_l_ecs[0] - ovoid_l_ecs[2]
        v_ovoid_l /= np.linalg.norm(v_ovoid_l)

        def angle_between(u, v):
            dot_val = np.clip(np.dot(u, v), -1.0, 1.0)
            return np.degrees(np.arccos(dot_val))

        angle_r = angle_between(v_tandem, v_ovoid_r)
        angle_l = angle_between(v_tandem, v_ovoid_l)
        angle_mean = 0.5 * (angle_r + angle_l)

        tandem_angle = angle_mean - 60.0

        allowed = np.array([0.0, 15.0, 30.0, 45.0])
        nominal_tandem_angle = allowed[np.argmin(np.abs(allowed - tandem_angle))]

        print(f"Likely tandem used = {nominal_tandem_angle:.0f}°")
    else:
        print("⚠️ Need ≥3 dwell points in each ovoid and ≥2 in tandem.")
        sys.exit()
else:
    print("⚠️ Not enough catheters to compute tandem–ovoid angle.")
    sys.exit()

# Create surface points and 5 mm points lateral to ovoids
r_pts = ovoid_r_ecs[[1, 2]].copy()   # shape (2,3)
l_pts = ovoid_l_ecs[[1, 2]].copy()   # shape (2,3)

r_pts[:, 0] -= 1   # 1 cm to x negative for right ovoid
l_pts[:, 0] += 1    # 1 cm to x positive for left ovoid
pt_surface_t = np.vstack([r_pts, l_pts])

r_pts[:, 0] -= 0.5   # 1.5 cm to x negative for right ovoid
l_pts[:, 0] += 0.5   # 1.5 cm to x positive for left ovoid
pt_5mm_t = np.vstack([r_pts, l_pts])

pt_surface = inverse_transform_points(pt_surface_t)
pt_5mm = inverse_transform_points(pt_5mm_t)

# ==========================================================
""" Infer ovoid size from transformed geometry
    Assumptions:
    catheter 1 = right ovoid, catheter 2 = left ovoid
    x-coordinate ~ ± (ovoid_diameter / 2)
    Clamp to standard diameters 20 mm to 40 mm. """ 
# ==========================================================
print("\nDetermine tandem and ovoid by geometry:")
if len(channels_ecs) >= 2:
    centroids = [np.mean(c, axis=0) for c in channels_ecs[:2]]
    x_mean_abs = np.mean([abs(c[0]) for c in centroids])
    possible_sizes = [20, 25, 30, 35, 40]
    ovoid_size = min(possible_sizes, key=lambda s: abs(x_mean_abs * 10 - s / 2.0))
    print(f"Likely ovoid size = {ovoid_size} mm")
else:
    ovoid_size = None
    print("⚠️ Could not estimate ovoid size.")
    sys.exit()

# ==========================================================
""" Read ovoid geometry config and apply Y-Z correction

    The config file defines geometry in Elekta coordinate
    The following correct the ovoid hole geometry to transformed frame:
    y_actual = y_config / cos(nominal_tandem_angle - 15°)
    z_actual = y_actual / tan(angle_tandem_ovoid_deg) """
# ==========================================================
if ovoid_size is not None and nominal_tandem_angle is not None:
    with open(ovoid_config_path, "r") as f:
        ovoid_config = json.load(f)

    ovoid_key = f"ovoid_{int(ovoid_size)}mm"
    if ovoid_key not in ovoid_config:
        raise KeyError(f"Ovoid size '{ovoid_key}' not found in configuration file.")

    hole_dict = ovoid_config[ovoid_key]["coordinates"]
    angle_correction = math.radians(nominal_tandem_angle - 15.0)
    cos_factor = math.cos(angle_correction)

    hole_locs_ecs = {}
    for key, val in hole_dict.items():
        x, y, z = val
        x = x / 10 # convert to cm
        y = y / 10
        z = z / 10
        y_corr = y / cos_factor
        hole_locs_ecs[key] = [x, y_corr, z]
    print(f"\n✅ Loaded ovoid geometry: {ovoid_config[ovoid_key]['description']}")

    # stash for later use
    ovoid_config[ovoid_key]["coordinates_corrected"] = hole_locs_ecs
else:
    print("⚠️ Skipping ovoid library correction because size or angle is missing.")
    sys.exit()

angle_tandem_ovoid_rad = math.radians(angle_mean)  # true geometric angle from earlier
tan_factor = math.tan(angle_tandem_ovoid_rad)

hole_locs_ecs_corr = {}
for key, val in hole_locs_ecs.items():
    x, y_corr, z = val
    z_corr = y_corr / tan_factor if abs(tan_factor) > 1e-6 else z
    hole_locs_ecs_corr[key] = [x, y_corr, z_corr]
hole_locs_ecs = hole_locs_ecs_corr

# ==========================================================
""" Compute initial dose from input dwell time """
# ==========================================================
def pairwise_distance(pts, dwell_positions):
    diff = pts[:,None,:] - dwell_positions[None,:,:]
    return np.maximum(np.linalg.norm(diff, axis=2), 0.01)

def compute_dose_rate(pts, dwell_positions, Sk, Lambda):
    # point source dose calculation
    r = pairwise_distance(pts, dwell_positions)
    dose = Sk * Lambda / 3600 / (r ** 2)
    return dose

dose_pt_surface = compute_dose_rate(pt_surface, dwell_pos_to, Sk, Lambda) @ dwell_time_to
dose_pt_5mm = compute_dose_rate(pt_5mm, dwell_pos_to, Sk, Lambda) @ dwell_time_to

# ----------------------------------------------------------------------------
# Scale the ovoids dwell time to achieve surface dose and 5 mm dose limits
# ----------------------------------------------------------------------------
if any(dose_pt_surface > pt_surface_limit_bt) or any(dose_pt_5mm > pt_5mm_limit_bt):
    scale_factor_surface = dose_pt_surface / pt_surface_limit_bt
    scale_factor_5mm = dose_pt_5mm / pt_5mm_limit_bt
    scale_factor = max(scale_factor_surface.max(), scale_factor_5mm.max())

    global_ov_idx = [  # find all ovoids dwell indices in global dwell list
        global_dw_idx
        for (ch_num, _), global_dw_idx in dwell_lookup.items()
        if ch_num in [1, 2]
    ]
    dwell_time_to[global_ov_idx] /= scale_factor  # halve ovoid dwell times for surface dose calculation

# Double check ICRU rectovaginal dose after scaling
dose_rectovaginal = compute_dose_rate(pointRecto[np.newaxis, :], dwell_pos_to, Sk, Lambda) @ dwell_time_to
if any(dose_rectovaginal > rectovaginal_limit_bt):
    scale_factor_rv = dose_rectovaginal / rectovaginal_limit_bt
    dwell_time_to[dwell_lookup[(1, 0)]] /= scale_factor_rv  # further reduce ovoid dwell times
    dwell_time_to[dwell_lookup[(2, 0)]] /= scale_factor_rv  # further reduce ovoid dwell times

# # Now compute mask within 2 cm of HR-CTV for OAR dose optimization
# dist_map = ndi.distance_transform_edt(~hrctv_mask, sampling=vox_size)

# bladder_mask_near_ctv = np.logical_and(bladder_mask, dist_map <= 2.0)
# return_mask_near_ctv = np.logical_and(rectum_mask, dist_map <= 2.0)
# sigmoid_mask_near_ctv = np.logical_and(sigmoid_mask, dist_map <= 2.0)
# bowel_mask_near_ctv = np.logical_and(bowel_mask, dist_map <= 2.0)

# oars = [
#     {"name": "bladder", "mask": bladder_mask_near_ctv, "limit": bladder_limit_bt},
#     {"name": "rectum",  "mask": return_mask_near_ctv,  "limit": rectum_limit_bt},
#     {"name": "sigmoid", "mask": sigmoid_mask_near_ctv, "limit": sigmoid_limit_bt},
#     {"name": "bowel",   "mask": bowel_mask_near_ctv,   "limit": bowel_limit_bt}
# ]

# -----------------------------------------
# Compute dose for each OAR
# -----------------------------------------
def compute_d2cc(dose, dose_limit, vox_size):
    n2cc = int(np.ceil(2.0 / np.prod(vox_size))) # compute d2cc voxel volume
    hotspot_idx = np.where(dose > dose_limit)[0] # Voxels over limit
    sorted_idx = np.argsort(-dose) # Sort dose descending to find hottest 2cc

    if len(sorted_idx) >= n2cc:
        d2cc = dose[sorted_idx][n2cc-1]
    else:
        d2cc = dose[sorted_idx][-1]

    hotspot_2cc_idx = sorted_idx[:min(n2cc, len(sorted_idx))] # hottest 2cc voxels
    hotspot_ex2cc_idx = np.setdiff1d(hotspot_idx, hotspot_2cc_idx, assume_unique=False) # Over limit but not in hottest 2cc
    return d2cc, hotspot_idx, hotspot_2cc_idx, hotspot_ex2cc_idx

n_dwell = len(dwell_time_to)
A_constr = []   # rows of A from all OARs
b_constr = []   # corresponding RHS values
# oar_info = {}        # to report D2cc before/after

for oar in oars.values():
    name  = oar["name"]
    mask  = oar["mask"]
    goal = oar["goal"]

    idx = np.argwhere(mask)
    if idx.size == 0: # no voxels in mask, skipping
        continue
    vox_pts = origin + idx * vox_size          # (N_vox, 3) in cm

    # Influence matrix for this OAR    
    A_oar = compute_dose_rate(vox_pts, dwell_pos_to, Sk, Lambda) # (N_vox, N_dwell)
    dose_oar = A_oar @ dwell_time_to # initial dose for this OAR (N_vox,)

    # Get oar d2cc
    d2cc, hotspot_idx, _, hotspot_ex2cc_idx = compute_d2cc(dose_oar, goal, vox_size)

    # Store info for later reporting
    oars[name]["vox_pts"] = vox_pts
    oars[name]["A_matrix"] = A_oar
    oars[name]["opt_idx"] = hotspot_idx
    oars[name]["d2cc"] = d2cc
    #     "A": A_oar,
    #     "dose": dose_oar,
    #     "pts": vox_pts,
    #     "limit": limit,
    #     "hotspot": hotspot_idx,
    # }

    # Build constraint rows for these extra hotspots
    if len(hotspot_ex2cc_idx) > 0:
        A_constr.append(A_oar[hotspot_ex2cc_idx, :])
        b_constr.append(goal - dose_oar[hotspot_ex2cc_idx])

# -----------------------------------------------------------
# Assemble constraint matrix and solve QP in Δt = t - t0
# -----------------------------------------------------------
print("\nSolve QP to reduce OAR dose in library plan:")
if len(A_constr) == 0:
    print("No hotspots over limits in any OAR; no adjustment needed.")
else:
    A_constr = np.vstack(A_constr)           # (M_total, N_dwell)
    b_constr = np.concatenate(b_constr)      # (M_total,)

    # Objective: minimal total change in dwell times
    dt = cp.Variable(n_dwell)   # dwell time changes
    objective = cp.Minimize(cp.sum_squares(dt))

    constraints = []
    constraints.append(A_constr @ dt <= b_constr) # Dose constraints: A_constr @ dt <= (limit - dose_oar)
    constraints.append(dt >= -dwell_time_to) # Bounds: -t0 <= dt <= 0  (only reduce, never negative)
    constraints.append(dt <= 0)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print("Warning: QP did not find an optimal solution; keeping original dwell times.")
    else:
        dwell_time_to = dwell_time_to + dt.value # update dwell time in T&O

# -----------------------------------------
# Recompute dose & D2cc for each OAR
# -----------------------------------------
for oar in oars.values():
    name = oar["name"]

    A_oar = oar["A_matrix"]

    dose_new = A_oar @ dwell_time_to
    n2cc = int(np.ceil(2.0 / np.prod(vox_size)))

    # D2cc before & after
    doses_new_sorted = np.sort(dose_new)[::-1]

    if len(doses_new_sorted) >= n2cc:
        D2cc_new = doses_new_sorted[n2cc-1]
    else:
        D2cc_new = doses_new_sorted[-1]
    d2cc_old = oar["d2cc"]

    print(f"\n{name} D2cc before: {d2cc_old:.3f}")
    print(f"{name} D2cc after:   {D2cc_new:.3f}")