# Self-inductance of an Archimedean spiral coil via Biot-Savart and magnetic energy.
# Cross-section: 5 um (width) x 1 um (height). J is uniform along the spiral tangent.
# Spiral: r(theta) = R_start + (R_end - R_start) * theta / theta_max, theta in [0, 2*pi*N_turns].
# U = integral B^2/(2*mu0) dV in the box; L = 2*U/I^2. Singularity: max(|R|, cutoff) in denominator.

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MU0 = 4.0 * math.pi * 1e-7


@dataclass(frozen=True)
class CoilParams:
    r_start_m: float = 3e-3
    r_end_m: float = 3.5e-3
    width_m: float = 5e-6
    height_m: float = 1e-6
    n_turns: float = 10.0
    current_a: float = 1.0
    n_theta: int = 400
    n_width: int = 5
    n_height: int = 2


@dataclass(frozen=True)
class DomainParams:
    box_mm: float = 20.0
    n_grid: int = 33
    cutoff_m: float | None = None
    # Skip air cells whose center is within this distance of any source (None = off).
    exclude_near_conductor_m: float | None = None


def build_coil_volume_elements(
    params: CoilParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discretize conductor into small prisms. Returns centers (N,3), J (N,3), volume (N,)."""
    w, h = params.width_m, params.height_m
    cross_area = w * h
    j_mag = params.current_a / cross_area

    theta_max = 2.0 * math.pi * params.n_turns
    thetas = np.linspace(0.0, theta_max, params.n_theta, endpoint=False)
    dtheta = theta_max / params.n_theta

    dr_dtheta = (params.r_end_m - params.r_start_m) / theta_max
    r = params.r_start_m + dr_dtheta * thetas

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    x0 = r * cos_t
    y0 = r * sin_t
    z0 = np.zeros_like(x0)

    dx_dtheta = dr_dtheta * cos_t - r * sin_t
    dy_dtheta = dr_dtheta * sin_t + r * cos_t
    t = np.stack([dx_dtheta, dy_dtheta, np.zeros_like(dx_dtheta)], axis=1)
    t_norm = np.linalg.norm(t, axis=1, keepdims=True)
    t_hat = t / np.maximum(t_norm, 1e-30)

    u = np.stack([-t_hat[:, 1], t_hat[:, 0], np.zeros(params.n_theta)], axis=1)
    u_norm = np.linalg.norm(u, axis=1, keepdims=True)
    u_hat = u / np.maximum(u_norm, 1e-30)

    dw = w / params.n_width
    dh = h / params.n_height

    centers: list[np.ndarray] = []
    tangents: list[np.ndarray] = []
    vols: list[float] = []

    for i in range(params.n_theta):
        ds = float(np.linalg.norm(t[i])) * dtheta
        vol_cell = dw * dh * ds
        if vol_cell <= 0.0:
            continue
        for iw in range(params.n_width):
            for ih in range(params.n_height):
                off_w = (iw + 0.5 - 0.5 * params.n_width) * dw
                off_h = (ih + 0.5 - 0.5 * params.n_height) * dh
                cx = x0[i] + off_w * u_hat[i, 0]
                cy = y0[i] + off_w * u_hat[i, 1]
                cz = z0[i] + off_h
                centers.append(np.array([cx, cy, cz], dtype=np.float64))
                tangents.append(t_hat[i].copy())
                vols.append(vol_cell)

    positions = np.stack(centers, axis=0)
    t_stack = np.stack(tangents, axis=0)
    j_vecs = j_mag * t_stack
    vol_arr = np.asarray(vols, dtype=np.float64)
    return positions, j_vecs, vol_arr


def biot_savart_chunk(
    obs: np.ndarray,
    src_pos: np.ndarray,
    j_vec: np.ndarray,
    vol: np.ndarray,
    cutoff: float,
    src_block: int = 256,
) -> np.ndarray:
    """obs (M,3); returns B (M,3). Volume-current Biot-Savart sum."""
    mu0_4pi = MU0 / (4.0 * math.pi)
    b = np.zeros_like(obs)
    for s0 in range(0, src_pos.shape[0], src_block):
        s1 = min(s0 + src_block, src_pos.shape[0])
        r_src = src_pos[s0:s1]
        j = j_vec[s0:s1]
        v = vol[s0:s1, None]
        d = obs[:, None, :] - r_src[None, :, :]
        dist = np.linalg.norm(d, axis=2, keepdims=True)
        dist_safe = np.maximum(dist, cutoff)
        cross = np.cross(j[None, :, :], d, axisa=2, axisb=2, axisc=2)
        b += np.sum(mu0_4pi * v[None, :, :] * cross / (dist_safe**3), axis=1)
    return b


def magnetic_energy_uniform_grid(
    src_pos: np.ndarray,
    j_vec: np.ndarray,
    vol_src: np.ndarray,
    box_half_m: float,
    n_grid: int,
    cutoff: float,
    z_chunk: int = 4,
    exclude_near_conductor_m: float | None = None,
) -> float:
    """Integrate B^2/(2*mu0) dV on a uniform cell-centered grid in [-box_half, box_half]^3.

    Cell-centered nodes reduce the chance of sampling exactly on the coil path (which
    inflates B^2). If exclude_near_conductor_m is set, skip cells whose center is closer
    than this distance to any source point (crude conductor neighborhood exclusion).
    """
    edges = np.linspace(-box_half_m, box_half_m, n_grid + 1)
    axes = 0.5 * (edges[:-1] + edges[1:])
    dx = float(axes[1] - axes[0])
    dcell = dx**3

    xs, ys, zs = np.meshgrid(axes, axes, axes, indexing="ij")
    total = 0.0

    for z0 in range(0, n_grid, z_chunk):
        z1 = min(z0 + z_chunk, n_grid)
        obs = np.stack(
            [
                xs[:, :, z0:z1].reshape(-1),
                ys[:, :, z0:z1].reshape(-1),
                zs[:, :, z0:z1].reshape(-1),
            ],
            axis=1,
        )
        if exclude_near_conductor_m is not None and exclude_near_conductor_m > 0.0:
            # distance from each obs to nearest source (chunked over sources for memory)
            dmin = np.full(obs.shape[0], np.inf)
            bs = 512
            for s0 in range(0, src_pos.shape[0], bs):
                s1 = min(s0 + bs, src_pos.shape[0])
                d = np.linalg.norm(obs[:, None, :] - src_pos[None, s0:s1, :], axis=2)
                dmin = np.minimum(dmin, np.min(d, axis=1))
            mask = dmin >= exclude_near_conductor_m
            if not np.any(mask):
                continue
            obs = obs[mask]

        b = biot_savart_chunk(obs, src_pos, j_vec, vol_src, cutoff)
        b2 = np.sum(b * b, axis=1)
        total += float(np.sum(b2) * dcell)

    return total / (2.0 * MU0)


def archimedean_turns_from_radial_pitch(
    r_start_m: float,
    r_end_m: float,
    pitch_center_to_center_m: float,
) -> float:
    """N_turns = (R_end - R_start) / pitch (center-to-center spacing per turn)."""
    dr = r_end_m - r_start_m
    if pitch_center_to_center_m <= 0.0:
        raise ValueError("pitch must be positive")
    return dr / pitch_center_to_center_m


def compute_B_axis_and_radial_plane(
    coil: CoilParams,
    z_axis_m: np.ndarray,
    r_radial_m: np.ndarray,
    z_plane_m: float = 0.0,
    cutoff_m: float | None = None,
    src_block: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """B on z-axis (0,0,z), and B on ray (r,0,z_plane) in a plane z = z_plane_m.

    For planar volume currents with J_z=0 and all sources in z≈0, any observation
    point with z_plane_m=0 lies in the same plane as J, hence Bx=By=0 exactly and
    only Bz is nonzero (J×r is along z). Use small |z_plane_m|>0 for cylindrical
    Br = sqrt(Bx^2+By^2) in the coil neighborhood.

    Returns (B_axis, B_radial), each (M,3) and (N,3) in Tesla.
    """
    src_pos, j_vec, vol_arr = build_coil_volume_elements(coil)
    cutoff = cutoff_m if cutoff_m is not None else 0.5 * max(coil.width_m, coil.height_m)

    obs_axis = np.stack(
        [np.zeros_like(z_axis_m), np.zeros_like(z_axis_m), z_axis_m],
        axis=1,
    )
    obs_rad = np.stack(
        [
            r_radial_m,
            np.zeros_like(r_radial_m),
            np.full_like(r_radial_m, z_plane_m, dtype=np.float64),
        ],
        axis=1,
    )

    b_axis = biot_savart_chunk(obs_axis, src_pos, j_vec, vol_arr, cutoff, src_block=src_block)
    b_rad = biot_savart_chunk(obs_rad, src_pos, j_vec, vol_arr, cutoff, src_block=src_block)
    return b_axis, b_rad


def compute_inductance(
    coil: CoilParams,
    domain: DomainParams,
) -> dict[str, float]:
    src_pos, j_vec, vol_arr = build_coil_volume_elements(coil)
    box_half = 0.5 * (domain.box_mm * 1e-3)
    cutoff = domain.cutoff_m
    if cutoff is None:
        cutoff = 0.5 * max(coil.width_m, coil.height_m)
    t0 = time.perf_counter()
    u_mag = magnetic_energy_uniform_grid(
        src_pos,
        j_vec,
        vol_arr,
        box_half,
        domain.n_grid,
        cutoff,
        exclude_near_conductor_m=domain.exclude_near_conductor_m,
    )
    i_a = coil.current_a
    l_h = 2.0 * u_mag / (i_a * i_a) if i_a != 0 else float("nan")
    elapsed = time.perf_counter() - t0
    return {
        "U_J": u_mag,
        "L_H": l_h,
        "L_nH": l_h * 1e9,
        "n_sources": float(src_pos.shape[0]),
        "cutoff_m": cutoff,
        "seconds": elapsed,
    }


def run_field_study_default_output() -> None:
    """User geometry: 50 turns (or ~10 um pitch = 5 um gap + 5 um width), I = 1 mA."""
    r0, r1 = 3e-3, 3.5e-3
    width, height = 5e-6, 1e-6
    # 5 um gap between adjacent turns (edge-to-edge) + 5 um wire width => 10 um pitch.
    pitch_cc = width + 5e-6
    n_turns = archimedean_turns_from_radial_pitch(r0, r1, pitch_cc)
    # Along-wire resolution: aim ~80 um arc length per segment at mean radius.
    theta_max = 2.0 * math.pi * n_turns
    r_mean = 0.5 * (r0 + r1)
    ds_target = 80e-6
    n_theta = max(2000, int(theta_max * r_mean / ds_target) + 1)

    coil = CoilParams(
        r_start_m=r0,
        r_end_m=r1,
        width_m=width,
        height_m=height,
        n_turns=n_turns,
        current_a=1e-3,
        n_theta=n_theta,
        n_width=5,
        n_height=1,
    )

    z_axis = np.linspace(-1e-3, 1e-3, 401)
    # 径向磁场只计算到线圈内侧附近（2.9 mm），不扫到外径以外
    r_rad = np.linspace(0.0, 2.9e-3, 801)
    cutoff = 0.5 * max(coil.width_m, coil.height_m)

    src_pos, j_vec, vol_arr = build_coil_volume_elements(coil)
    obs_axis = np.stack(
        [np.zeros_like(z_axis), np.zeros_like(z_axis), z_axis],
        axis=1,
    )
    obs_rad_z0 = np.stack([r_rad, np.zeros_like(r_rad), np.zeros_like(r_rad)], axis=1)
    z_probe = 0.5 * coil.height_m + 0.5e-6
    obs_rad_off = np.stack(
        [r_rad, np.zeros_like(r_rad), np.full_like(r_rad, z_probe, dtype=np.float64)],
        axis=1,
    )

    t0 = time.perf_counter()
    b_axis = biot_savart_chunk(obs_axis, src_pos, j_vec, vol_arr, cutoff)
    b_rad_z0 = biot_savart_chunk(obs_rad_z0, src_pos, j_vec, vol_arr, cutoff)
    b_rad_off = biot_savart_chunk(obs_rad_off, src_pos, j_vec, vol_arr, cutoff)
    elapsed = time.perf_counter() - t0

    out_dir = Path(__file__).resolve().parent
    brho0 = np.sqrt(b_rad_z0[:, 0] ** 2 + b_rad_z0[:, 1] ** 2)
    brho1 = np.sqrt(b_rad_off[:, 0] ** 2 + b_rad_off[:, 1] ** 2)
    np.savetxt(
        out_dir / "B_axis_z.csv",
        np.column_stack([z_axis, b_axis[:, 0], b_axis[:, 1], b_axis[:, 2]]),
        delimiter=",",
        header="z_m,Bx_T,By_T,Bz_T",
        comments="",
    )
    np.savetxt(
        out_dir / "B_radial_line_z0.csv",
        np.column_stack([r_rad, b_rad_z0[:, 0], b_rad_z0[:, 1], b_rad_z0[:, 2], brho0]),
        delimiter=",",
        header="r_m,Bx_T,By_T,Bz_T,Brho_xy_T",
        comments="",
    )
    np.savetxt(
        out_dir / "B_radial_line_z_probe.csv",
        np.column_stack(
            [r_rad, b_rad_off[:, 0], b_rad_off[:, 1], b_rad_off[:, 2], brho1]
        ),
        delimiter=",",
        header="r_m,Bx_T,By_T,Bz_T,Brho_xy_T_z_probe",
        comments="",
    )

    iz0 = int(np.argmin(np.abs(z_axis)))
    ir0 = int(np.argmin(np.abs(r_rad)))

    print("=== Archimedean spiral magnetic field (Biot-Savart, uniform J) ===")
    print(f"R_start={r0*1e3:.3f} mm, R_end={r1*1e3:.3f} mm, cross-section {width*1e6:.0f}x{height*1e6:.0f} um")
    print(f"Turns N = {n_turns:.4f} (pitch cc = {pitch_cc*1e6:.1f} um => 5 um gap + 5 um width)")
    print(f"I = {coil.current_a*1e3:.3f} mA, n_theta={coil.n_theta}, n_width={coil.n_width}, n_height={coil.n_height}")
    print(f"Volume sources = {src_pos.shape[0]}, cutoff = {cutoff*1e6:.2f} um")
    print(f"CPU time fields: {elapsed:.2f} s")
    print(
        f"B at center (0,0,0): Bz={b_rad_z0[ir0, 2]:.6e} T; "
        f"on-axis same z: Bz={b_axis[iz0, 2]:.6e} T"
    )
    print(
        "Note: for strictly z=0 in the coil plane, J lies in xy => Bx=By=0 exactly; "
        f"see Brho_xy in B_radial_line_z0.csv (zeros). Nonzero in-plane Brho at z_probe="
        f"{z_probe*1e6:.3f} um (half conductor height + 0.5 um): see B_radial_line_z_probe.csv."
    )
    print(
        f"Wrote {out_dir / 'B_axis_z.csv'}, {out_dir / 'B_radial_line_z0.csv'}, "
        f"{out_dir / 'B_radial_line_z_probe.csv'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Archimedean spiral inductance via Biot-Savart and magnetic energy.",
    )
    parser.add_argument(
        "--fields",
        action="store_true",
        help="Compute on-axis and in-plane radial B for default user geometry; write CSV.",
    )
    parser.add_argument("--n-turns", type=float, default=10.0)
    parser.add_argument("--n-grid", type=int, default=33)
    parser.add_argument("--n-theta", type=int, default=400)
    parser.add_argument("--current", type=float, default=1.0)
    parser.add_argument("--cutoff-um", type=float, default=None)
    parser.add_argument(
        "--exclude-near-um",
        type=float,
        default=None,
        help="If set, skip B^2 integration for cells whose center is within this "
        "distance of any source point (approx. exclude conductor volume).",
    )
    args = parser.parse_args()

    if args.fields:
        run_field_study_default_output()
        return

    coil = CoilParams(
        n_turns=args.n_turns,
        n_theta=args.n_theta,
        current_a=args.current,
    )
    cutoff_m = None if args.cutoff_um is None else args.cutoff_um * 1e-6
    ex_m = None if args.exclude_near_um is None else args.exclude_near_um * 1e-6
    domain = DomainParams(
        n_grid=args.n_grid,
        cutoff_m=cutoff_m,
        exclude_near_conductor_m=ex_m,
    )

    cutoff = domain.cutoff_m if domain.cutoff_m is not None else 0.5 * max(
        coil.width_m, coil.height_m
    )

    print("geometry / discretization")
    out = compute_inductance(coil, domain)
    print(f"  volume elements: {int(out['n_sources'])}")
    print(
        f"  box: {domain.box_mm} mm, grid {domain.n_grid}^3, "
        f"cutoff = {cutoff * 1e6:.3f} um"
    )
    print("results")
    print(f"  U = {out['U_J']:.6e} J  (I = {coil.current_a} A)")
    print(f"  L = 2U/I^2 = {out['L_H']:.6e} H = {out['L_nH']:.4f} nH")
    print(f"  time: {out['seconds']:.2f} s")


if __name__ == "__main__":
    main()
