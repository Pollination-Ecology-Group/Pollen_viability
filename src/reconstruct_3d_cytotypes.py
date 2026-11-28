#!/usr/bin/env python3
"""
reconstruct_3d_cytotypes.py — Pollen_latitude 2-stage 3D stereological & mesh correction
════════════════════════════════════════════════════════════════════════════════════════
Applies the exact two-stage 3D reconstruction algorithm from `Pollen_latitude`
(`src/reconstruct_3d_species.py`) to each ploidy cytotype (`ory_2x`, `nat_2x`,
`nat_4x`, `neo_4x`) in `Pollen_viability`:

Stage 1 — Stereological 3D axis & orientation unfolding:
  Fits 4 nested physical 3D projection models (`sphere`, `prolate`, `oblate`,
  `triaxial`) with viscous/gravitational slide-settling parameter `tau` in [0, 3.5],
  biological grain-size SD `sigma_g`, and 2D contour axis noise `sigma_e` via
  24-angle 2D Sliced Wasserstein Distance across all hydrated isolated viable grains,
  with slide-level block-bootstrap SEs (n_boot = 32) on (p_BA, q_CA, k_ell_3d).

Stage 2 — Non-ellipsoidal 3D surface mesh from S3 grain outlines:
  Samples polar (<= Q25 aspect ratio) and equatorial (>= Q75 aspect ratio) 2D
  polygons from S3 `_det.json` files across top slides per cytotype, aligns Fourier
  3-fold (tricolpate) polar and equatorial silhouettes, constructs a closed genus-0
  Wavefront `.obj` mesh, and computes exact divergence-theorem 3D volume, surface
  area, Wadell 3D sphericity, non-ellipsoidal factor `phi_mesh`, 3D grain volumes
  (`viable_vol_3d_median_um3`, `viable_vol_3d_mean_um3`), and 3D-corrected
  volume-equivalent spherical diameter (`viable_diam_3d_um`).

Outputs:
  - outcomes/pollen_3d_cytotype_models.csv
  - outcomes/meshes_3d/{cytotype}.obj  (+ outcomes/pollen_3d_meshes.zip)
  - Updates outcomes/pollen_viability_slides_final.csv & outcomes/pollen_viability_plants_final.csv
    (and output/qc/final_viability.csv) with 3D-corrected diameter and volume columns.
"""

import concurrent.futures
import io
import json
import math
import os
import sys
import time
import zipfile
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pollen_size_analyzer import make_s3, S3_BUCKET, TILE_PREFIX, DEFAULT_UM_PER_PX  # noqa: E402

CYTOTYPES = ["ory_2x", "nat_2x", "nat_4x", "neo_4x"]
TILE_PX = 640
CONF = 0.50
MAX_SLIDES_PER_CYT = 16
MAX_POLYS_PER_STRATUM = 160
N_THETA = 45
N_PHI = 72

# ── Deterministic quasi-Monte Carlo orientation & noise grids (exact Pollen_latitude) ──
M_ORI = 600
N_RESAMP = 1024

_idx = np.arange(M_ORI)
_vz = (_idx + 0.5) / M_ORI
_phi = 2.0 * np.pi * _idx / ((1.0 + np.sqrt(5.0)) / 2.0)
_vxy = np.sqrt(np.maximum(0.0, 1.0 - _vz**2))
_vx, _vy = _vxy * np.cos(_phi), _vxy * np.sin(_phi)
VX2, VY2, VZ2 = _vx[:, None] ** 2, _vy[:, None] ** 2, _vz[:, None] ** 2

_rng = np.random.default_rng(20261112)
G_STD = _rng.standard_normal(N_RESAMP)
E1_STD = _rng.standard_normal(N_RESAMP)
E2_STD = _rng.standard_normal(N_RESAMP)
U_STRAT = (np.arange(N_RESAMP) + 0.5) / N_RESAMP

SWD_ANGLES = np.linspace(0.0, np.pi, 24, endpoint=False)
PROJ = np.vstack([np.cos(SWD_ANGLES), np.sin(SWD_ANGLES)])
Q_GRID = np.linspace(0.02, 0.98, 65)
Q_IDX = np.clip(np.round(Q_GRID * (N_RESAMP - 1)).astype(int), 0, N_RESAMP - 1)


def simulate_projections(p, q, tau, sg, se):
    p2, q2 = p * p, q * q
    h2 = VX2[:, 0] + p2 * VY2[:, 0] + q2 * VZ2[:, 0]
    denom = max(1.0 - q2, 1e-6)
    e_norm = (h2 - q2) / denom
    lw = -tau * e_norm
    lw -= lw.max()
    w = np.exp(lw)
    w /= w.sum()

    tr = (1.0 + p2 + q2) - h2
    det = p2 * q2 * VX2[:, 0] + q2 * VY2[:, 0] + p2 * VZ2[:, 0]
    disc = np.sqrt(np.maximum(0.0, tr * tr - 4.0 * det))
    l1 = np.maximum(1e-8, 0.5 * (tr + disc))
    l2 = np.maximum(1e-8, 0.5 * (tr - disc))
    mu_bar = 0.25 * np.log(l1 * l2)
    delta = 0.25 * np.log(l1 / l2)

    cw = np.cumsum(w)
    cw[-1] = 1.0
    picks = np.clip(np.searchsorted(cw, U_STRAT), 0, M_ORI - 1)

    u = mu_bar[picks] + sg * G_STD
    d_obs = np.sqrt((delta[picks] + se * E1_STD) ** 2 + (se * E2_STD) ** 2)
    u_med = float(np.median(u))
    d_sim_med = math.exp(u_med)
    d_ori = np.exp(mu_bar)
    d3_ori_rat = float(np.sum(w * (d_ori / d_sim_med) ** 3))
    return np.vstack([u - u_med, d_obs]), d_sim_med, d3_ori_rat


def filter_hydrated_grains(df):
    if len(df) < 20:
        return df, 0.0
    maj_med = float(np.median(df["rmaj"]))
    maj_mad = max(float(np.median(np.abs(df["rmaj"] - maj_med))) * 1.4826, 0.03)
    ar_med = float(np.median(df["aspect_ratio"]))
    shriveled = (df["rmaj"] < maj_med - 2.2 * maj_mad) & (df["aspect_ratio"] > max(1.15, ar_med))
    u_lo, u_hi = np.quantile(df["u"], [0.003, 0.997])
    d_hi = np.quantile(df["delta"], 0.997)
    extreme = (~df["u"].between(u_lo, u_hi)) | (df["delta"] > d_hi)
    drop = shriveled | extreme
    if (~drop).sum() < 15:
        return df, 0.0
    return df[~drop], float(shriveled.mean())


def _pe_label(pe):
    if pe < 0.50:
        return "peroblate"
    if pe < 0.75:
        return "oblate"
    if pe < 0.88:
        return "suboblate"
    if pe < 0.99:
        return "oblate-spheroidal"
    if pe <= 1.01:
        return "spheroidal"
    if pe <= 1.14:
        return "prolate-spheroidal"
    if pe <= 1.33:
        return "subprolate"
    if pe <= 2.00:
        return "prolate"
    return "perprolate"


def erdtman_class(kind, p, q):
    if kind == "sphere":
        return "spheroidal"
    if kind == "prolate":
        return _pe_label(1.0 / max(q, 1e-6))
    if kind == "oblate":
        return _pe_label(q)
    if (1.0 - p) < 0.30 * (1.0 - q):
        return "triaxial-" + _pe_label(q / max(0.5 * (1.0 + p), 1e-6))
    if (p - q) < 0.30 * (1.0 - q):
        return "triaxial-" + _pe_label(1.0 / max(0.5 * (p + q), 1e-6))
    return "triaxial"


def fit_stereological_3d(u_obs, d_obs, slide_ids=None, n_boot=32):
    if len(u_obs) > 8000:
        rng = np.random.default_rng(12345)
        take = rng.choice(len(u_obs), 8000, replace=False)
        u_fit, d_fit = u_obs[take], d_obs[take]
        s_fit = slide_ids[take] if slide_ids is not None else None
    else:
        u_fit, d_fit, s_fit = u_obs, d_obs, slide_ids

    def make_emp_q(u_arr, d_arr):
        proj = PROJ[0, :, None] * u_arr[None, :] + PROJ[1, :, None] * d_arr[None, :]
        return np.quantile(proj, Q_GRID, axis=1).T

    emp_q = make_emp_q(u_fit, d_fit)

    def swd(p, q, tau, sg, se, target_q=emp_q):
        pts, _, _ = simulate_projections(p, q, tau, sg, se)
        sp = np.sort(PROJ.T @ pts, axis=1)
        return float(np.mean(np.abs(sp[:, Q_IDX] - target_q)))

    reg_tau = lambda tau: 4e-4 * (tau / 3.5) ** 2

    r_sph = differential_evolution(
        lambda x: swd(1.0, 1.0, 0.0, x[0], x[1]),
        [(0.008, 0.28), (0.002, 0.14)],
        seed=42, maxiter=25, popsize=8, polish=True,
    )
    r_pro = differential_evolution(
        lambda x: swd(x[0], x[0], x[1], x[2], x[3]) + reg_tau(x[1]),
        [(0.25, 1.0), (0.0, 3.5), (0.008, 0.28), (0.002, 0.14)],
        seed=42, maxiter=35, popsize=9, polish=True,
    )
    r_obl = differential_evolution(
        lambda x: swd(1.0, x[0], x[1], x[2], x[3]) + reg_tau(x[1]),
        [(0.25, 1.0), (0.0, 3.5), (0.008, 0.28), (0.002, 0.14)],
        seed=42, maxiter=35, popsize=9, polish=True,
    )
    r_tri = differential_evolution(
        lambda x: swd(x[0], x[0] * x[1], x[2], x[3], x[4]) + reg_tau(x[2]),
        [(0.25, 1.0), (0.25, 1.0), (0.0, 3.5), (0.008, 0.28), (0.002, 0.14)],
        seed=42, maxiter=40, popsize=10, polish=True,
    )

    l_sph = swd(1.0, 1.0, 0.0, *r_sph.x)
    l_pro = swd(r_pro.x[0], r_pro.x[0], r_pro.x[1], r_pro.x[2], r_pro.x[3])
    l_obl = swd(1.0, r_obl.x[0], r_obl.x[1], r_obl.x[2], r_obl.x[3])
    l_tri = swd(r_tri.x[0], r_tri.x[0] * r_tri.x[1], r_tri.x[2], r_tri.x[3], r_tri.x[4])

    if l_pro <= l_obl:
        sph_kind, l_spheroid, sph_par = "prolate", l_pro, (r_pro.x[0], r_pro.x[0], r_pro.x[1], r_pro.x[2], r_pro.x[3])
    else:
        sph_kind, l_spheroid, sph_par = "oblate", l_obl, (1.0, r_obl.x[0], r_obl.x[1], r_obl.x[2], r_obl.x[3])

    if (l_sph - min(l_spheroid, l_tri)) < max(0.0007, 0.08 * l_sph):
        kind, best_loss, par = "sphere", l_sph, (1.0, 1.0, 0.0, float(r_sph.x[0]), float(r_sph.x[1]))
    elif (l_spheroid - l_tri) >= max(0.0007, 0.08 * l_spheroid):
        kind, best_loss, par = "triaxial", l_tri, (
            float(r_tri.x[0]), float(r_tri.x[0] * r_tri.x[1]), float(r_tri.x[2]), float(r_tri.x[3]), float(r_tri.x[4])
        )
    else:
        kind, best_loss, par = sph_kind, l_spheroid, tuple(float(v) for v in sph_par)

    p, q, tau, sg, se = par
    _, d_sim_med, d3_ori_rat = simulate_projections(p, q, tau, sg, se)
    k_ell = (p * q) / (d_sim_med ** 3)

    boot_k, boot_p, boot_q = [], [], []
    rng = np.random.default_rng(777)
    unique_slides = np.unique(s_fit) if s_fit is not None else np.array([])
    for _ in range(n_boot):
        if len(unique_slides) >= 4:
            drawn = rng.choice(unique_slides, len(unique_slides), replace=True)
            idx_b = np.concatenate([np.flatnonzero(s_fit == sl) for sl in drawn])
        else:
            idx_b = rng.choice(len(u_fit), len(u_fit), replace=True)
        if len(idx_b) < 15:
            continue
        bq = make_emp_q(u_fit[idx_b], d_fit[idx_b])
        if kind == "sphere":
            boot_k.append(1.0); boot_p.append(1.0); boot_q.append(1.0)
        elif kind == "prolate":
            rb = minimize(
                lambda x: swd(np.clip(x[0], 0.25, 1.0), np.clip(x[0], 0.25, 1.0), np.clip(x[1], 0.0, 3.5),
                              np.clip(x[2], 0.005, 0.3), np.clip(x[3], 0.002, 0.15), bq) + reg_tau(np.clip(x[1], 0.0, 3.5)),
                [q, tau, sg, se], method="Nelder-Mead", options={"maxiter": 80},
            )
            qb = float(np.clip(rb.x[0], 0.25, 1.0))
            tb = float(np.clip(rb.x[1], 0.0, 3.5))
            sgb, seb = float(np.clip(rb.x[2], 0.005, 0.3)), float(np.clip(rb.x[3], 0.002, 0.15))
            _, dsm, _ = simulate_projections(qb, qb, tb, sgb, seb)
            boot_p.append(qb); boot_q.append(qb); boot_k.append((qb * qb) / (dsm ** 3))
        elif kind == "oblate":
            rb = minimize(
                lambda x: swd(1.0, np.clip(x[0], 0.25, 1.0), np.clip(x[1], 0.0, 3.5),
                              np.clip(x[2], 0.005, 0.3), np.clip(x[3], 0.002, 0.15), bq) + reg_tau(np.clip(x[1], 0.0, 3.5)),
                [q, tau, sg, se], method="Nelder-Mead", options={"maxiter": 80},
            )
            qb = float(np.clip(rb.x[0], 0.25, 1.0))
            tb = float(np.clip(rb.x[1], 0.0, 3.5))
            sgb, seb = float(np.clip(rb.x[2], 0.005, 0.3)), float(np.clip(rb.x[3], 0.002, 0.15))
            _, dsm, _ = simulate_projections(1.0, qb, tb, sgb, seb)
            boot_p.append(1.0); boot_q.append(qb); boot_k.append(qb / (dsm ** 3))
        else:
            rq0 = q / max(p, 1e-6)
            rb = minimize(
                lambda x: swd(np.clip(x[0], 0.25, 1.0), np.clip(x[0], 0.25, 1.0) * np.clip(x[1], 0.25, 1.0),
                              np.clip(x[2], 0.0, 3.5), np.clip(x[3], 0.005, 0.3), np.clip(x[4], 0.002, 0.15), bq)
                          + reg_tau(np.clip(x[2], 0.0, 3.5)),
                [p, rq0, tau, sg, se], method="Nelder-Mead", options={"maxiter": 100},
            )
            pb = float(np.clip(rb.x[0], 0.25, 1.0))
            qb = pb * float(np.clip(rb.x[1], 0.25, 1.0))
            tb = float(np.clip(rb.x[2], 0.0, 3.5))
            sgb, seb = float(np.clip(rb.x[3], 0.005, 0.3)), float(np.clip(rb.x[4], 0.002, 0.15))
            _, dsm, _ = simulate_projections(pb, qb, tb, sgb, seb)
            boot_p.append(pb); boot_q.append(qb); boot_k.append((pb * qb) / (dsm ** 3))

    return {
        "model_3d": kind,
        "shape_class_erdtman": erdtman_class(kind, p, q),
        "p_BA": p,
        "q_CA": q,
        "tau_settling": tau,
        "sigma_grain_size": sg,
        "sigma_axis_noise": se,
        "d_sim_med": d_sim_med,
        "d3_ori_rat": d3_ori_rat,
        "k_ell_3d": k_ell,
        "k_ell_3d_se": float(np.std(boot_k, ddof=1)) if len(boot_k) > 1 else 0.0,
        "p_BA_se": float(np.std(boot_p, ddof=1)) if len(boot_p) > 1 else 0.0,
        "q_CA_se": float(np.std(boot_q, ddof=1)) if len(boot_q) > 1 else 0.0,
        "swd_loss": best_loss,
        "swd_sphere": l_sph,
        "swd_prolate": l_pro,
        "swd_oblate": l_obl,
        "swd_triaxial": l_tri,
    }


def align_and_profile_polygons(polys, mode="polar"):
    phi_grid = np.linspace(0.0, 2.0 * np.pi, N_PHI, endpoint=False)
    if len(polys) < 8:
        return np.ones(N_PHI), 0.0

    radial_stack = []
    c3_over_c2 = []
    for p in polys:
        if len(p) < 6:
            continue
        c = p - p.mean(axis=0, keepdims=True)
        area = abs(cv2.contourArea(c.astype(np.float32).reshape(-1, 1, 2)))
        if area <= 1e-3:
            continue
        r_eq = math.sqrt(area / math.pi)
        c = c / r_eq

        pad = 4
        scale_px = 40.0
        pts_px = c * scale_px
        x0, y0 = np.floor(pts_px.min(axis=0)).astype(int) - pad
        x1, y1 = np.ceil(pts_px.max(axis=0)).astype(int) + pad
        if (x1 - x0) > 600 or (y1 - y0) > 600 or (x1 <= x0) or (y1 <= y0):
            continue
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(mask, [np.round(pts_px - [x0, y0]).astype(np.int32)], 1)
        ys, xs = np.nonzero(mask)
        if len(xs) < 20:
            continue
        mxy = np.vstack([xs, ys]).astype(float)
        mxy -= mxy.mean(axis=1, keepdims=True)
        evals, evecs = np.linalg.eigh(np.cov(mxy, bias=True))
        ratio = math.sqrt(evals[1] / evals[0]) if evals[0] > 0 else 1.0
        a_fit = math.sqrt(ratio)
        b_fit = 1.0 / math.sqrt(ratio)

        centroid_um = np.array([xs.mean() + x0, ys.mean() + y0]) / scale_px
        c_centered = c - centroid_um
        rot = np.vstack([evecs[:, 1], evecs[:, 0]])
        if np.linalg.det(rot) < 0:
            rot[1] *= -1
        al = rot @ c_centered.T

        m_al = rot @ mxy
        if np.mean(m_al[0] ** 3) < 0:
            al[0] *= -1
        if np.mean(m_al[1] ** 3) < 0:
            al[1] *= -1

        diffs = np.diff(np.hstack([al, al[:, :1]]), axis=1)
        seg = np.hypot(diffs[0], diffs[1])
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        if cum[-1] <= 0:
            continue
        u_arc = np.linspace(0.0, cum[-1], 256, endpoint=False)
        ax = np.interp(u_arc, cum, np.append(al[0], al[0, 0]))
        ay = np.interp(u_arc, cum, np.append(al[1], al[1, 0]))

        tx, ty = ax / a_fit, ay / b_fit
        ang = np.mod(np.arctan2(ty, tx), 2.0 * np.pi)
        rad = np.hypot(tx, ty)
        order = np.argsort(ang)
        ang_ext = np.concatenate([ang[order] - 2 * np.pi, ang[order], ang[order] + 2 * np.pi])
        rad_ext = np.tile(rad[order], 3)
        rho = np.interp(phi_grid, ang_ext, rad_ext)

        if mode == "polar":
            z2 = np.mean(rho * np.exp(-1j * 2 * phi_grid))
            z3 = np.mean(rho * np.exp(-1j * 3 * phi_grid))
            c3_over_c2.append(abs(z3) / max(abs(z2), 1e-4))
            if abs(z3) > abs(z2) and abs(z3) > 0.015:
                psi3 = np.angle(z3) / 3.0
                ang3 = np.mod(ang - psi3, 2.0 * np.pi)
                o3 = np.argsort(ang3)
                rho = np.interp(phi_grid, np.concatenate([ang3[o3] - 2 * np.pi, ang3[o3], ang3[o3] + 2 * np.pi]), np.tile(rad[o3], 3))

        rho /= math.sqrt(float(np.mean(rho ** 2)))
        radial_stack.append(rho)

    if not radial_stack:
        return np.ones(N_PHI), 0.0
    rho_med = np.median(np.vstack(radial_stack), axis=0)
    spec = np.fft.rfft(rho_med)
    spec[11:] = 0
    rho_smooth = np.fft.irfft(spec, n=N_PHI)
    rho_smooth = np.clip(rho_smooth, 0.6, 1.4)
    rho_smooth /= math.sqrt(float(np.mean(rho_smooth ** 2)))
    return rho_smooth, (float(np.median(c3_over_c2)) if c3_over_c2 else 0.0)


def _interp_periodic(angles, rho_vec):
    phis = np.linspace(0.0, 2.0 * np.pi, len(rho_vec), endpoint=False)
    ext_p = np.concatenate([phis - 2 * np.pi, phis, phis + 2 * np.pi])
    ext_r = np.tile(rho_vec, 3)
    return np.interp(np.mod(angles, 2.0 * np.pi), ext_p, ext_r)


def mesh_volume_and_area(V, F):
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * float(np.sum(np.linalg.norm(cross, axis=1)))
    vol = abs(float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))))) / 6.0
    return vol, area


def _unit_ellipsoid_mesh_vol(A, B, C, F, thetas, phis, z_polar):
    ct, st = np.cos(thetas)[:, None], np.sin(thetas)[:, None]
    cp, sp = np.cos(phis)[None, :], np.sin(phis)[None, :]
    if z_polar:
        X, Y, Z = (A / 2) * st * cp, (B / 2) * st * sp, (C / 2) * ct * np.ones_like(cp)
        vn, vs = [[0, 0, C / 2]], [[0, 0, -C / 2]]
    else:
        X, Y, Z = (A / 2) * ct * np.ones_like(cp), (B / 2) * st * cp, (C / 2) * st * sp
        vn, vs = [[A / 2, 0, 0]], [[-A / 2, 0, 0]]
    V = np.vstack([vn, np.stack([X, Y, Z], axis=-1).reshape(-1, 3), vs])
    vol, _ = mesh_volume_and_area(V, F)
    return vol


def build_closed_mesh(A_um, B_um, C_um, kind, rho_polar, rho_equat):
    thetas = np.linspace(0.0, np.pi, N_THETA + 2)[1:-1]
    phis = np.linspace(0.0, 2.0 * np.pi, N_PHI, endpoint=False)
    cos_t, sin_t = np.cos(thetas)[:, None], np.sin(thetas)[:, None]
    cos_p, sin_p = np.cos(phis)[None, :], np.sin(phis)[None, :]
    rho_p = rho_polar[None, :]

    if kind == "sphere":
        X = (A_um / 2.0) * sin_t * rho_p * cos_p
        Y = (B_um / 2.0) * sin_t * rho_p * sin_p
        Z = (C_um / 2.0) * cos_t * np.ones_like(cos_p)
        v_north = np.array([[0.0, 0.0, C_um / 2.0]])
        v_south = np.array([[0.0, 0.0, -C_um / 2.0]])
    elif kind == "oblate":
        r_mer = 0.5 * (_interp_periodic(0.5 * np.pi - thetas, rho_equat) + _interp_periodic(0.5 * np.pi + thetas, rho_equat))[:, None]
        r_pn = float(_interp_periodic(0.5 * np.pi, rho_equat))
        r_ps = float(_interp_periodic(1.5 * np.pi, rho_equat))
        X = (A_um / 2.0) * r_mer * sin_t * rho_p * cos_p
        Y = (B_um / 2.0) * r_mer * sin_t * rho_p * sin_p
        Z = (C_um / 2.0) * r_mer * cos_t * np.ones_like(cos_p)
        v_north = np.array([[0.0, 0.0, (C_um / 2.0) * r_pn]])
        v_south = np.array([[0.0, 0.0, -(C_um / 2.0) * r_ps]])
    else:
        r_plus = _interp_periodic(thetas, rho_equat)[:, None]
        r_minus = _interp_periodic(2.0 * np.pi - thetas, rho_equat)[:, None]
        r_sym = 0.5 * (r_plus + r_minus)
        r_blend = 0.5 * (1.0 + cos_p) * r_plus + 0.5 * (1.0 - cos_p) * r_minus
        r_pn = float(_interp_periodic(0.0, rho_equat))
        r_ps = float(_interp_periodic(np.pi, rho_equat))
        X = (A_um / 2.0) * r_sym * cos_t * np.ones_like(cos_p)
        Y = (B_um / 2.0) * r_blend * rho_p * sin_t * cos_p
        Z = (C_um / 2.0) * r_sym * rho_p * sin_t * sin_p
        v_north = np.array([[(A_um / 2.0) * r_pn, 0.0, 0.0]])
        v_south = np.array([[-(A_um / 2.0) * r_ps, 0.0, 0.0]])

    rings = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    V = np.vstack([v_north, rings, v_south])
    idx_south = len(V) - 1

    faces = []
    for j in range(N_PHI):
        faces.append((0, 1 + j, 1 + (j + 1) % N_PHI))
    for i in range(N_THETA - 1):
        r0 = 1 + i * N_PHI
        r1 = 1 + (i + 1) * N_PHI
        for j in range(N_PHI):
            jn = (j + 1) % N_PHI
            faces.append((r0 + j, r1 + j, r1 + jn))
            faces.append((r0 + j, r1 + jn, r0 + jn))
    rlast = 1 + (N_THETA - 1) * N_PHI
    for j in range(N_PHI):
        faces.append((idx_south, rlast + (j + 1) % N_PHI, rlast + j))
    F = np.asarray(faces, dtype=np.int32)

    vol, area = mesh_volume_and_area(V, F)
    V_ref = _unit_ellipsoid_mesh_vol(A_um, B_um, C_um, F, thetas, phis, kind in ("oblate", "sphere"))
    ell_exact = (math.pi / 6.0) * A_um * B_um * C_um
    phi_mesh = float(np.clip(vol / max(V_ref, 1e-9), 0.65, 1.35))
    vol_corrected = ell_exact * phi_mesh
    sphericity = (math.pi ** (1.0 / 3.0) * (6.0 * vol_corrected) ** (2.0 / 3.0)) / max(area, 1e-9)
    return V, F, vol_corrected, area, float(np.clip(sphericity, 0.0, 1.0)), phi_mesh


def format_obj(V, F, header_comment):
    lines = [f"# {header_comment}"]
    for x, y, z in V:
        lines.append(f"v {x:.4f} {y:.4f} {z:.4f}")
    for i, j, k in F:
        lines.append(f"f {i + 1} {j + 1} {k + 1}")
    return "\n".join(lines) + "\n"


def sample_polygons_from_s3(s3, sub_df):
    """Fetch 2D polygons (in micrometres) for polar (<= Q25 AR) and equatorial (>= Q75 AR) grains from S3 `_det.json`."""
    q25, q75 = np.quantile(sub_df["aspect_ratio"], [0.25, 0.75])
    top_slides = sub_df["slide_name"].value_counts().head(MAX_SLIDES_PER_CYT).index.tolist()
    low_s = sub_df[(sub_df["slide_name"].isin(top_slides)) & (sub_df["aspect_ratio"] <= q25)]
    high_s = sub_df[(sub_df["slide_name"].isin(top_slides)) & (sub_df["aspect_ratio"] >= q75)]
    if len(low_s) > MAX_POLYS_PER_STRATUM:
        low_s = low_s.sample(MAX_POLYS_PER_STRATUM, random_state=42)
    if len(high_s) > MAX_POLYS_PER_STRATUM:
        high_s = high_s.sample(MAX_POLYS_PER_STRATUM, random_state=43)

    # Index det keys for top_slides
    det_by_slide = {}
    def _list_slide(sname):
        pfx = f"{TILE_PREFIX.rstrip('/')}/{sname}/"
        dets = []
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET, Prefix=pfx):
            for o in page.get("Contents", []):
                k = o["Key"]
                if k.endswith("_det.json") and "_tile_" in k:
                    ys, xs = k[:-9].rsplit("_tile_", 1)[1].split("_", 1)
                    dets.append((int(xs), int(ys), k))
        return sname, dets

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        for sname, dets in ex.map(_list_slide, top_slides):
            det_by_slide[sname] = dets

    def _fetch_polys(target_df):
        tile_targets = defaultdict(list)
        for _, r in target_df.iterrows():
            cx, cy = float(r["center_x_px"]), float(r["center_y_px"])
            best_k, best_d = None, 1e18
            for ox, oy, k in det_by_slide.get(r["slide_name"], []):
                lx, ly = cx - ox, cy - oy
                if 20 <= lx <= 620 and 20 <= ly <= 620:
                    d = (lx - 320) ** 2 + (ly - 320) ** 2
                    if d < best_d:
                        best_d, best_k = d, (k, ox, oy, lx, ly)
            if best_k:
                tile_targets[(best_k[0], best_k[1], best_k[2])].append((best_k[3], best_k[4]))

        def _read_tile(item):
            (k, ox, oy), pts_local = item
            out = []
            try:
                det = json.loads(s3.get_object(Bucket=S3_BUCKET, Key=k)["Body"].read())
                boxes, masks = det.get("boxes", []), det.get("masks_xyn", [])
                for lx, ly in pts_local:
                    best_i, best_d = -1, 1e9
                    for i, b in enumerate(boxes):
                        if float(b[4]) < CONF or int(b[5]) != 0:
                            continue
                        gx, gy = 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])
                        d = math.hypot(gx - lx, gy - ly)
                        if d < best_d:
                            best_d, best_i = d, i
                    if best_i >= 0 and best_d <= 8.0 and best_i < len(masks) and len(masks[best_i]) >= 6:
                        p_px = np.array(masks[best_i], dtype=float) * TILE_PX
                        if p_px[:, 0].min() > 3 and p_px[:, 1].min() > 3 and p_px[:, 0].max() < 637 and p_px[:, 1].max() < 637:
                            out.append(p_px * DEFAULT_UM_PER_PX)
            except Exception:
                pass
            return out

        polys = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
            for res in ex.map(_read_tile, list(tile_targets.items())):
                polys.extend(res)
        return polys

    return _fetch_polys(low_s), _fetch_polys(high_s)


def main():
    t0 = time.time()
    s3 = make_s3()

    slides = pd.read_csv("outcomes/pollen_viability_slides_final.csv")
    fv_all = pd.read_csv("output/qc/final_viability.csv")
    md_df = pd.read_csv("output/qc/class_min_diam.csv")[["cytotype", "class_name", "min_diam_um"]]
    ct_df = pd.read_csv("output/qc/class_conf_threshold.csv")[["cytotype", "class_name", "conf_threshold"]]
    grains_raw = pd.read_csv("output/size_analysis/pollen_grain_measurements.csv")
    focus_gz = "output/qc/grain_focus_flags.csv.gz"
    if os.path.exists(focus_gz):
        gf = pd.read_csv(focus_gz, usecols=["slide_name", "grain_id", "norm_sharpness"])
        grains_raw = grains_raw.merge(gf, on=["slide_name", "grain_id"], how="left")
        grains_raw = grains_raw[grains_raw["norm_sharpness"].fillna(6.0) >= 3.50]

    g = grains_raw.merge(md_df, on=["cytotype", "class_name"], how="left").merge(
        ct_df, on=["cytotype", "class_name"], how="left"
    )
    g = g[
        (g["valid_for_size"] == 1)
        & (g["slide_name"].isin(set(slides["slide_name"])))
        & (g["class_name"] == "viable")
        & (g["min_diam_um"].isna() | (g["diameter_um"] >= g["min_diam_um"]))
        & (g["conf_threshold"].isna() | (g["conf"] >= g["conf_threshold"]))
        & (g["major_axis_um"] > 0)
        & (g["minor_axis_um"] > 0)
    ].copy()

    # Exact Pollen_latitude stereological coordinates u and delta relative to slide median d_eq
    g["d_eq"] = g["diameter_um"]
    smed = g.groupby("slide_name")["d_eq"].transform("median")
    g["rmaj"] = g["major_axis_um"] / smed
    g["rmin"] = g["minor_axis_um"] / smed
    g["u"] = np.log(g["d_eq"] / smed)
    g["delta"] = 0.5 * np.log(np.maximum(1.0, g["major_axis_um"] / g["minor_axis_um"]))

    os.makedirs("outcomes/meshes_3d", exist_ok=True)
    rows = []
    mesh_bytes = {}

    for cyt in CYTOTYPES:
        ts = time.time()
        sub_raw = g[g["cytotype"] == cyt]
        sub, shriv = filter_hydrated_grains(sub_raw)
        n_slides = int(sub["slide_name"].nunique())
        diam_cyt = float(slides[slides["cytotype"] == cyt]["viable_median_diam_um"].median())

        fit = fit_stereological_3d(
            sub["u"].to_numpy(),
            sub["delta"].to_numpy(),
            slide_ids=sub["slide_name"].to_numpy(),
            n_boot=32,
        )

        A_um = diam_cyt / fit["d_sim_med"]
        B_um = A_um * fit["p_BA"]
        C_um = A_um * fit["q_CA"]

        low_polys, high_polys = sample_polygons_from_s3(s3, sub)
        if fit["model_3d"] == "sphere":
            rho_polar, tri_idx = align_and_profile_polygons(low_polys + high_polys, mode="polar")
            rho_equat = np.ones(N_PHI)
        else:
            rho_polar, tri_idx = align_and_profile_polygons(low_polys, mode="polar")
            rho_equat, _ = align_and_profile_polygons(high_polys, mode="equatorial")

        V, F, vol_mesh_med, surf_area_um2, sphericity_3d, phi_mesh = build_closed_mesh(
            A_um, B_um, C_um, fit["model_3d"], rho_polar, rho_equat
        )
        vol_ell_med = (math.pi / 6.0) * A_um * B_um * C_um
        vol_2d_sphere = (math.pi / 6.0) * (diam_cyt ** 3)
        k_total_vol = fit["k_ell_3d"] * phi_mesh
        k_diam_3d = k_total_vol ** (1.0 / 3.0)
        diam_3d_med = diam_cyt * k_diam_3d

        obj_txt = format_obj(
            V, F,
            f"{cyt} | model={fit['model_3d']} ({fit['shape_class_erdtman']}) | axes=({A_um:.2f}, {B_um:.2f}, {C_um:.2f}) um | V_3d={vol_mesh_med:.1f} um3 | D_3d={diam_3d_med:.2f} um",
        )
        obj_path = f"outcomes/meshes_3d/{cyt}.obj"
        with open(obj_path, "w", encoding="utf-8") as f_obj:
            f_obj.write(obj_txt)
        mesh_bytes[cyt] = obj_txt.encode("utf-8")

        rows.append({
            "cytotype": cyt,
            "n_usable_slides": n_slides,
            "n_measured_grains": int(len(sub_raw)),
            "n_hydrated_grains": int(len(sub)),
            "shriveled_grain_share": round(shriv, 4),
            "n_polar_polygons": len(low_polys),
            "n_equatorial_polygons": len(high_polys),
            "model_3d": fit["model_3d"],
            "shape_class_erdtman": fit["shape_class_erdtman"],
            "Diam_2d_median_um": round(diam_cyt, 3),
            "Diam_3d_corrected_um": round(diam_3d_med, 3),
            "Axis_A_um": round(A_um, 3),
            "Axis_B_um": round(B_um, 3),
            "Axis_C_um": round(C_um, 3),
            "ratio_BA": round(fit["p_BA"], 4),
            "ratio_BA_se": round(fit["p_BA_se"], 4),
            "ratio_CA": round(fit["q_CA"], 4),
            "ratio_CA_se": round(fit["q_CA_se"], 4),
            "tau_settling": round(fit["tau_settling"], 3),
            "sigma_grain_size": round(fit["sigma_grain_size"], 4),
            "sigma_axis_noise": round(fit["sigma_axis_noise"], 4),
            "d_sim_med": round(fit["d_sim_med"], 5),
            "d3_ori_rat": round(fit["d3_ori_rat"], 5),
            "k_ell_3d": round(fit["k_ell_3d"], 5),
            "k_ell_3d_se": round(fit["k_ell_3d_se"], 5),
            "phi_mesh": round(phi_mesh, 4),
            "k_vol_3d_total": round(k_total_vol, 5),
            "k_diam_3d": round(k_diam_3d, 5),
            "tri_symmetry_index": round(tri_idx, 4),
            "Surf_area_3d_um2": round(surf_area_um2, 2),
            "Sphericity_3d": round(sphericity_3d, 4),
            "Vol_2d_sphere_um3": round(vol_2d_sphere, 2),
            "Vol_3d_ell_canonical_um3": round(vol_ell_med, 2),
            "Vol_3d_mesh_canonical_um3": round(vol_mesh_med, 2),
            "swd_loss": round(fit["swd_loss"], 5),
            "swd_sphere": round(fit["swd_sphere"], 5),
            "swd_prolate": round(fit["swd_prolate"], 5),
            "swd_oblate": round(fit["swd_oblate"], 5),
            "swd_triaxial": round(fit["swd_triaxial"], 5),
        })
        print(
            f"✓ {cyt:7s}: {fit['model_3d']} ({fit['shape_class_erdtman']}) | axes=({A_um:.2f}, {B_um:.2f}, {C_um:.2f}) um | "
            f"tau={fit['tau_settling']:.2f} | k_ell={fit['k_ell_3d']:.4f}±{fit['k_ell_3d_se']:.4f} | phi_mesh={phi_mesh:.4f} | "
            f"D_2d={diam_cyt:.2f} -> D_3d={diam_3d_med:.2f} um | V_3d={vol_mesh_med:.1f} um3 ({time.time()-ts:.1f}s)"
        )

    m3d = pd.DataFrame(rows)
    m3d.to_csv("outcomes/pollen_3d_cytotype_models.csv", index=False)

    with zipfile.ZipFile("outcomes/pollen_3d_meshes.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for cyt, b in mesh_bytes.items():
            zf.writestr(f"meshes/{cyt}.obj", b)

    # Apply Pollen_latitude 3D volume & diameter correction onto slide-level and plant-level tables
    g["d3_rel"] = (g["d_eq"] / smed) ** 3
    slide_disp = g.groupby("slide_name")["d3_rel"].mean().clip(0.8, 2.5)
    c_par = m3d.set_index("cytotype")

    def add_3d_cols(df_in):
        d = df_in.copy()
        d["model_3d"] = d["cytotype"].map(c_par["model_3d"])
        d["shape_class_erdtman"] = d["cytotype"].map(c_par["shape_class_erdtman"])
        k_ell = d["cytotype"].map(c_par["k_ell_3d"])
        phi_m = d["cytotype"].map(c_par["phi_mesh"])
        d3_ori = d["cytotype"].map(c_par["d3_ori_rat"])
        disp = d["slide_name"].map(slide_disp).fillna(1.0)

        k_tot = k_ell * phi_m
        d["viable_diam_3d_um"] = (d["viable_median_diam_um"] * (k_tot ** (1.0 / 3.0))).round(3)
        v_sph = (np.pi / 6.0) * (d["viable_median_diam_um"] ** 3)
        d["viable_vol_3d_ell_median_um3"] = (v_sph * k_ell).round(2)
        d["viable_vol_3d_mesh_median_um3"] = (v_sph * k_tot).round(2)
        bio_disp = (disp / d3_ori.replace(0, np.nan)).clip(1.0, 2.2).fillna(1.0)
        d["viable_vol_3d_mesh_mean_um3"] = (d["viable_vol_3d_mesh_median_um3"] * bio_disp).round(2)
        d["Vol_3d_viable_total_slide_um3"] = (d["viable_vol_3d_mesh_mean_um3"] * d["viable_count"]).round(1)
        return d

    fv_updated = add_3d_cols(fv_all)
    fv_updated.to_csv("output/qc/final_viability.csv", index=False)

    slides_updated = add_3d_cols(slides)
    slides_updated.to_csv("outcomes/pollen_viability_slides_final.csv", index=False)

    # Rebuild plant-level summary including 3D diameter and 3D volume columns
    plants_old = pd.read_csv("outcomes/pollen_viability_plants_final.csv")
    p_3d = slides_updated.groupby("plant_id").agg(
        viable_diam_3d_um=("viable_diam_3d_um", "mean"),
        viable_vol_3d_mesh_median_um3=("viable_vol_3d_mesh_median_um3", "mean"),
        viable_vol_3d_mesh_mean_um3=("viable_vol_3d_mesh_mean_um3", "mean"),
        Vol_3d_viable_total_slide_um3=("Vol_3d_viable_total_slide_um3", "mean"),
    ).round(2).reset_index()
    for col in ("viable_diam_3d_um", "viable_vol_3d_mesh_median_um3", "viable_vol_3d_mesh_mean_um3", "Vol_3d_viable_total_slide_um3"):
        if col in plants_old.columns:
            plants_old = plants_old.drop(columns=[col])
    plants_updated = plants_old.merge(p_3d, on="plant_id", how="left")
    plants_updated.to_csv("outcomes/pollen_viability_plants_final.csv", index=False)

    print(f"\n🏁 Completed 3D stereological & mesh reconstruction across all 4 cytotypes in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()

