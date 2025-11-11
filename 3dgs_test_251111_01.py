#!/usr/bin/env python3
"""
Minimal PyTorch Gaussian Splatting–style dose reconstruction from
DICOM RTPLAN dwell positions with axial/coronal/sagittal supervision.

Quick start (conda):
  conda create -n dosegs python=3.10 -y
  conda activate dosegs
  pip install torch pydicom numpy tqdm matplotlib
  # Optional acceleration (not required for this minimal version):
  # pip install gsplat   # https://github.com/nerfstudio-project/gsplat

Run:
  python 3DGS_dose_from_dwell_minimal.py \
    --ct_dir /path/to/CT_series_dir \
    --rtplan /path/to/RTPLAN.dcm \
    --rtdose /path/to/RTDOSE.dcm \
    --out_dir ./out

This script:
  • Reads CT geometry to build world↔voxel transforms (DICOM LPS).
  • Extracts dwell positions (world, mm) from RTPLAN (Control Point 3D Position).
  • Initializes a set of 3D isotropic/elliptic Gaussians around dwell tracks.
  • Builds three orthogonal supervision planes (ax, co, sa) from RTDOSE.
  • Optimizes Gaussian parameters (μ, σ, w) to fit those planes.
  • Saves preview PNGs and a NumPy volume of the reconstructed dose.

Notes:
  • This is a research prototype. Tag names vary by vendor—robustness guards are included
    but you may need to adapt extract_dwells().
  • Physics prior (TG‑43‑like) is sketched; you can plug your tables to refine weights.
  • For true GS rasterization, swap gaussian_field() with a gsplat-based renderer.
"""
from __future__ import annotations
import os
import argparse
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pydicom
from pydicom.dataset import Dataset
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------
# DICOM utilities
# -----------------------

def load_ct_series(ct_dir: str) -> Dict:
    """Load one CT series folder (sorted by ImagePositionPatient along slice axis).
    Returns geometry and minimal info; does not load voxels (not needed for this script).
    """
    files = [pydicom.dcmread(str(Path(ct_dir)/f)) for f in os.listdir(ct_dir) if f.lower().endswith('.dcm')]
    if not files:
        raise FileNotFoundError("No DICOM files in ct_dir")
    # Group by InstanceNumber or ImagePositionPatient (safer)
    files.sort(key=lambda d: (float(d.ImagePositionPatient[2]) if 'ImagePositionPatient' in d else d.InstanceNumber))
    ref = files[0]
    ps = np.array(ref.PixelSpacing, dtype=float)  # (row, col) mm
    # DICOM uses LPS coordinate frame. Orientation is two direction cosines (row, col); slice is cross.
    ori = np.array(ref.ImageOrientationPatient, dtype=float).reshape(2, 3)
    row_dir, col_dir = ori[0], ori[1]
    slice_dir = np.cross(row_dir, col_dir)
    origin = np.array(ref.ImagePositionPatient, dtype=float)
    # Estimate slice spacing from adjacent positions.
    if len(files) > 1 and 'ImagePositionPatient' in files[1]:
        dz = np.dot(np.array(files[1].ImagePositionPatient) - origin, slice_dir)
    else:
        dz = float(getattr(ref, 'SliceThickness', 1.0))
    spacing = np.array([ps[1], ps[0], dz], dtype=float)  # x,y,z spacing in mm along col,row,slice axes
    shape_ij = (int(ref.Rows), int(ref.Columns))
    num_slices = len(files)
    shape_zyx = (num_slices, shape_ij[0], shape_ij[1])
    return dict(
        origin=origin,           # world LPS (mm)
        row_dir=row_dir,
        col_dir=col_dir,
        slice_dir=slice_dir,
        spacing=spacing,         # (dx,dy,dz) in world-aligned basis vectors above
        shape_zyx=shape_zyx,     # (Z, Y, X)
        series_uid=getattr(ref, 'SeriesInstanceUID', 'unknown')
    )


def load_rtdose(rtdose_path: str) -> Dict:
    ds = pydicom.dcmread(rtdose_path)
    dose_grid_scaling = float(ds.DoseGridScaling)
    arr = ds.pixel_array.astype(np.float32) * dose_grid_scaling  # Gy
    # Geometry for RTDOSE (can differ from CT grid)
    ori = np.array(ds.ImageOrientationPatient, dtype=float).reshape(2, 3)
    row_dir, col_dir = ori[0], ori[1]
    slice_dir = np.cross(row_dir, col_dir)
    origin = np.array(ds.ImagePositionPatient, dtype=float)
    ps = np.array(ds.PixelSpacing, dtype=float)  # (row, col)
    # GridFrameOffsetVector gives per-slice offsets along slice_dir
    if hasattr(ds, 'GridFrameOffsetVector'):
        offsets = np.array(ds.GridFrameOffsetVector, dtype=float)
    else:
        # Fallback to SliceThickness
        offsets = np.arange(arr.shape[0]) * float(getattr(ds, 'SliceThickness', 1.0))
    spacing = np.array([ps[1], ps[0], np.mean(np.diff(offsets)) if len(offsets) > 1 else offsets[0]], dtype=float)
    return dict(
        dose=arr,                # (Z,Y,X) Gy
        origin=origin,
        row_dir=row_dir,
        col_dir=col_dir,
        slice_dir=slice_dir,
        spacing=spacing,         # (dx,dy,dz) in mm
        offsets=offsets,
        shape_zyx=arr.shape
    )


def extract_dwells(rtplan_path: str) -> List[Dict]:
    """Extract dwell positions and channel directions from RTPLAN.
    Returns list of dicts: {positions: [N,3] mm (world LPS), direction: [3] (unit, if inferable)}
    """
    ds = pydicom.dcmread(rtplan_path)
    dwells = []
    # IEC: Brachy Application Setup Sequence (300A,00A0) -> Channel Sequence (300A,0280)
    # if not hasattr(ds, 'BrachyApplicationSetupSequence'):
    #     raise ValueError('RTPLAN missing BrachyApplicationSetupSequence')
    for app in ds.BrachyApplicationSetupSequence:
        if not hasattr(app, 'ChannelSequence'):
            continue
        for ch in app.ChannelSequence:
            cps = getattr(ch, 'BrachyControlPointSequence', None) or getattr(ch, 'ControlPointSequence', None)
            if cps is None:
                continue
            pts = []
            for cp in cps:
                # Preferred: Control Point 3D Position (300A,02D4)
                if hasattr(cp, 'ControlPoint3DPosition'):
                    p = np.array(cp.ControlPoint3DPosition, dtype=float)
                    pts.append(p)
                # Some vendors store in (300A,012C) Source Applicator Position or similar—add fallbacks as needed.
                elif hasattr(cp, 'SourceApplicatorPosition') and hasattr(cp, 'TableTopLateralPosition'):
                    # This branch is vendor-specific; leave as TODO
                    pass
            if len(pts) >= 2:
                pts = np.stack(pts, axis=0)  # [N,3]
                # Infer channel direction from first-to-last control point
                ch_dir = pts[-1] - pts[0]
                n = np.linalg.norm(ch_dir) + 1e-8
                ch_dir = ch_dir / n
                dwells.append(dict(positions=pts, direction=ch_dir))
            elif len(pts) == 1:
                dwells.append(dict(positions=np.stack(pts), direction=np.array([0.0,0.0,1.0])))
    if not dwells:
        raise ValueError('No dwell control points found; check vendor tags or add fallbacks.')
    return dwells


# -----------------------
# Geometry / transforms
# -----------------------

def make_view_grids(rtdose_geo: Dict, plane: str, step: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build world coordinates for a supervision plane from RTDOSE grid.
    plane in {'ax','co','sa'}. Returns (grid_xyz [H,W,3], H, W)
    step: sub-sampling stride for speed.
    """
    Z, Y, X = rtdose_geo['shape_zyx']
    dx, dy, dz = rtdose_geo['spacing']
    o = rtdose_geo['origin']
    rd, cd, sd = rtdose_geo['row_dir'], rtdose_geo['col_dir'], rtdose_geo['slice_dir']
    # Build axes for each plane
    if plane == 'ax':
        # rows -> Y, cols -> X at a chosen slice (use mid-slice)
        zi = Z // 2
        rows = np.arange(0, Y, step)
        cols = np.arange(0, X, step)
        rr, cc = np.meshgrid(rows, cols, indexing='ij')
        # world = origin + cc*dx*cd + rr*dy*rd + offset_z*sd
        off = rtdose_geo['offsets'][zi] if len(rtdose_geo['offsets']) else zi*dz
        xyz = o + (cc*dx)[:, :, None]*cd + (rr*dy)[:, :, None]*rd + off*sd
    elif plane == 'co':
        yi = Y // 2
        zs = np.arange(0, Z, step)
        cols = np.arange(0, X, step)
        zz, cc = np.meshgrid(zs, cols, indexing='ij')
        off = (rtdose_geo['offsets'][zz] if len(rtdose_geo['offsets']) else (zz*dz))
        # Fix row index (yi)
        xyz = o + (cc*dx)[:, :, None]*cd + (yi*dy)*rd + (off[..., None])*sd
    elif plane == 'sa':
        xi = X // 2
        zs = np.arange(0, Z, step)
        rows = np.arange(0, Y, step)
        zz, rr = np.meshgrid(zs, rows, indexing='ij')
        off = (rtdose_geo['offsets'][zz] if len(rtdose_geo['offsets']) else (zz*dz))
        xyz = o + (xi*dx)*cd + (rr*dy)[:, :, None]*rd + (off[..., None])*sd
    else:
        raise ValueError('plane must be ax/co/sa')
    H, W = xyz.shape[:2]
    return xyz.reshape(-1, 3), H, W


# -----------------------
# Gaussian field + priors
# -----------------------

class GaussianField(nn.Module):
    def __init__(self, mus: torch.Tensor, sigmas: torch.Tensor, weights: torch.Tensor, anisotropy: torch.Tensor | None = None):
        super().__init__()
        # mus: [K,3] (world, mm)
        # sigmas: [K] (mm) or [K,3] for axis-wise; here isotropic scalar per component
        # weights: [K]
        self.mus = nn.Parameter(mus)            # [K,3]
        self.log_sigmas = nn.Parameter(torch.log(sigmas))  # [K]
        self.weights = nn.Parameter(weights)    # [K]
        # Optional: simple anisotropy via per‑Gaussian stretch factors along a unit direction (channel)
        if anisotropy is None:
            self.aniso = nn.Parameter(torch.zeros(mus.shape[0]))  # 0 => isotropic
        else:
            self.aniso = nn.Parameter(anisotropy)

    def forward(self, xq: torch.Tensor, dir_vec: torch.Tensor | None = None) -> torch.Tensor:
        """Evaluate scalar field at query points xq [Q,3].
        dir_vec (optional): [3] unit vector defining prolate stretch axis for all gaussians (channel axis).
        """
        mu = self.mus[None, :, :]          # [1,K,3]
        diff = xq[:, None, :] - mu         # [Q,K,3]
        r2 = (diff**2).sum(dim=-1)         # [Q,K]
        sigma = torch.exp(self.log_sigmas)[None, :]  # [1,K]
        if dir_vec is not None:
            # Prolate: shrink along channel axis depending on aniso factor (positive => thinner)
            v = dir_vec / (torch.norm(dir_vec) + 1e-8)
            proj = (diff @ v)              # [Q,K]
            r2 = r2 + self.aniso[None, :]*(proj**2)
        G = torch.exp(-0.5 * (r2 / (sigma**2 + 1e-9)))  # [Q,K]
        D = (G * self.weights[None, :]).sum(dim=1)      # [Q]
        return D


def tg43_like_weights(mus: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Very rough 1/r^2 prior around a reference center. Replace with tables if available."""
    r = np.linalg.norm(mus - center[None, :], axis=1) + 1e-3
    w = 1.0 / (r**2)
    w /= (w.max() + 1e-9)
    return w.astype(np.float32)


def init_gaussians_from_dwells(dwells: List[Dict], 
                               per_cp: int = 4,
                               sigma_mm: float = 2.0,
                               jitter_mm: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a small Gaussian cloud around each control point along each channel.
    Returns mus [K,3], sigmas [K], weights [K], aniso [K] (init ~0).
    """
    mus_list, sig_list, wei_list, aniso_list = [], [], [], []
    for ch in dwells:
        pts = ch['positions']  # [N,3]
        ch_dir = ch.get('direction', np.array([0,0,1], dtype=float))
        center = pts.mean(axis=0)
        for p in pts:
            # Per control point, spawn a few Gaussians with small jitter
            base = np.repeat(p[None, :], per_cp, axis=0)
            jitter = np.random.normal(scale=jitter_mm, size=base.shape)
            gpos = base + jitter
            mus_list.append(gpos)
            sig_list.append(np.full((per_cp,), sigma_mm, dtype=np.float32))
            # TG‑43‑like weights around channel center
            w = tg43_like_weights(gpos, center=center)
            wei_list.append(w)
            aniso_list.append(np.zeros((per_cp,), dtype=np.float32))
    mus = torch.from_numpy(np.concatenate(mus_list, axis=0)).float()
    sig = torch.from_numpy(np.concatenate(sig_list, axis=0)).float()
    wei = torch.from_numpy(np.concatenate(wei_list, axis=0)).float()
    aniso = torch.from_numpy(np.concatenate(aniso_list, axis=0)).float()
    # Normalize initial weights
    wei = wei / (wei.max() + 1e-6)
    return mus, sig, wei, aniso


# -----------------------
# Training
# -----------------------

def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    # Load geometry & supervision
    ct_geo = load_ct_series(args.ct_dir)
    dose_geo = load_rtdose(args.rtdose)
    dose = dose_geo['dose']  # (Z,Y,X)

    # Supervision planes (ax/co/sa)
    xyz_ax, Hax, Wax = make_view_grids(dose_geo, 'ax', step=args.sup_stride)
    xyz_co, Hco, Wco = make_view_grids(dose_geo, 'co', step=args.sup_stride)
    xyz_sa, Hsa, Wsa = make_view_grids(dose_geo, 'sa', step=args.sup_stride)

    # Ground-truth slices from RTDOSE (mid-slices)
    Z, Y, X = dose.shape
    dose_ax = dose[Z//2, ::args.sup_stride, ::args.sup_stride]
    dose_co = dose[::args.sup_stride, Y//2, ::args.sup_stride]
    dose_sa = dose[::args.sup_stride, ::args.sup_stride, X//2]

    # Dwells → initial Gaussians
    dwells = extract_dwells(args.rtplan)
    mus, sig, wei, aniso = init_gaussians_from_dwells(
        dwells, per_cp=args.per_cp, sigma_mm=args.sigma_mm, jitter_mm=args.jitter_mm)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    field = GaussianField(mus.to(device), sig.to(device), wei.to(device), aniso.to(device)).to(device)

    # Channel direction (use mean of channels; or per‑Gaussian mapping if you wish)
    if len(dwells) > 0:
        mean_dir = np.mean([d['direction'] for d in dwells], axis=0)
        dir_vec = torch.from_numpy(mean_dir.astype(np.float32)).to(device)
    else:
        dir_vec = torch.tensor([0.0,0.0,1.0], device=device)

    # Query tensors
    xq_ax = torch.from_numpy(xyz_ax.astype(np.float32)).to(device)
    xq_co = torch.from_numpy(xyz_co.astype(np.float32)).to(device)
    xq_sa = torch.from_numpy(xyz_sa.astype(np.float32)).to(device)
    gt_ax = torch.from_numpy(dose_ax.reshape(-1).astype(np.float32)).to(device)
    gt_co = torch.from_numpy(dose_co.reshape(-1).astype(np.float32)).to(device)
    gt_sa = torch.from_numpy(dose_sa.reshape(-1).astype(np.float32)).to(device)

    # Normalize Gy to [0,1] for stability
    scale = max(gt_ax.max(), gt_co.max(), gt_sa.max()).item() + 1e-6
    gt_ax /= scale; gt_co /= scale; gt_sa /= scale

    opt = torch.optim.Adam(field.parameters(), lr=args.lr)
    pbar = tqdm(range(args.iters), desc='Optimizing 3D Gaussian Field')

    for it in pbar:
        opt.zero_grad()
        pred_ax = field(xq_ax, dir_vec=dir_vec)
        pred_co = field(xq_co, dir_vec=dir_vec)
        pred_sa = field(xq_sa, dir_vec=dir_vec)
        # L2 loss across views + small smooth prior via weight decay implicit
        loss = ((pred_ax - gt_ax).pow(2).mean() +
                (pred_co - gt_co).pow(2).mean() +
                (pred_sa - gt_sa).pow(2).mean())
        loss.backward()
        opt.step()
        pbar.set_postfix(loss=float(loss))

        if (it+1) % args.vis_every == 0 or it == args.iters - 1:
            with torch.no_grad():
                for tag, pred, H, W in [('ax', pred_ax, Hax, Wax), ('co', pred_co, Hco, Wco), ('sa', pred_sa, Hsa, Wsa)]:
                    img = (pred.reshape(H, W).detach().cpu().numpy()) * scale
                    plt.figure(figsize=(5,5)); plt.imshow(img, origin='lower'); plt.title(f'{tag} pred (Gy) @ {it+1}')
                    plt.colorbar(); plt.tight_layout()
                    plt.savefig(os.path.join(args.out_dir, f'pred_{tag}_{it+1:05d}.png')); plt.close()

    # Final: export coarse 3D volume by sampling full RTDOSE grid
    with torch.no_grad():
        Z, Y, X = dose.shape
        dx, dy, dz = dose_geo['spacing']
        o = dose_geo['origin']; rd, cd, sd = dose_geo['row_dir'], dose_geo['col_dir'], dose_geo['slice_dir']
        vol = np.zeros_like(dose, dtype=np.float32)
        for zi in tqdm(range(Z), desc='Sampling 3D field'):
            off = (dose_geo['offsets'][zi] if len(dose_geo['offsets']) else zi*dz)
            rows = np.arange(0, Y)
            cols = np.arange(0, X)
            rr, cc = np.meshgrid(rows, cols, indexing='ij')
            xyz = o + (cc*dx)[:, :, None]*cd + (rr*dy)[:, :, None]*rd + off*sd
            xq = torch.from_numpy(xyz.reshape(-1,3).astype(np.float32)).to(device)
            pred = field(xq, dir_vec=dir_vec).reshape(Y, X).detach().cpu().numpy() * scale
            vol[zi] = pred
        np.save(os.path.join(args.out_dir, 'dose_recon.npy'), vol)
        # Quick orthogonal previews
        midz, midy, midx = Z//2, Y//2, X//2
        for tag, img in [('ax', vol[midz]), ('co', vol[:, midy, :]), ('sa', vol[:, :, midx])]:
            plt.figure(figsize=(5,5)); plt.imshow(img, origin='lower'); plt.title(f'final {tag} (Gy)')
            plt.colorbar(); plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, f'final_{tag}.png')); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ct_dir', type=str, required=True, help='Folder of one CT series (same FOR UID).')
    ap.add_argument('--rtplan', type=str, required=True, help='RTPLAN DICOM file.')
    ap.add_argument('--rtdose', type=str, required=True, help='RTDOSE DICOM file (used for supervision planes).')
    ap.add_argument('--out_dir', type=str, default='./out')
    ap.add_argument('--iters', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--sup_stride', type=int, default=2, help='Subsample factor for supervision planes (speed).')
    ap.add_argument('--per_cp', type=int, default=4, help='#Gaussians per control point.')
    ap.add_argument('--sigma_mm', type=float, default=2.0)
    ap.add_argument('--jitter_mm', type=float, default=1.0)
    ap.add_argument('--vis_every', type=int, default=200)
    args = ap.parse_args()
    run(args)

if __name__ == '__main__':
    main()
