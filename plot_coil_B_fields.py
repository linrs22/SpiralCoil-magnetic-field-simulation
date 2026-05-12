"""
读取 archimedean_spiral_inductance.py --fields 生成的 CSV，绘制三张图并保存为 JPG。
以坐标 0 处 $B_z$ 为 100%，其余为相对百分比；0 处磁场数值以 **Gs（高斯）** 标注（$1\\,\\mathrm{T}=10^4\\,\\mathrm{Gs}$）。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 中文显示（Windows 常见字体）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _load_csv(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"未找到数据文件: {path}，请先运行: python archimedean_spiral_inductance.py --fields")
    return np.loadtxt(path, delimiter=",", skiprows=1)


def _index_at_zero(x: np.ndarray) -> int:
    return int(np.argmin(np.abs(x)))


def _bz_percent_reference(bz: np.ndarray, i0: int) -> tuple[float, str]:
    """Returns (ref_T, note_suffix). ref used for Bz% and for Bx,By,Brho% when needed."""
    bz0 = float(bz[i0])
    if abs(bz0) > 1e-40:
        return bz0, ""
    m = float(np.nanmax(np.abs(bz)))
    if m > 1e-40:
        return m, "（$z=0$ 处 $B_z\\approx 0$，用 $|B_z|_\\mathrm{max}$ 作基准）"
    return 1.0, "（$B_z$ 全零，百分比无意义）"


def _format_bz_gs(bz_tesla: float) -> str:
    """Tesla -> Gauss (1 T = 10^4 Gs), compact string."""
    gs = bz_tesla * 1e4
    if abs(gs) >= 1e-2 or gs == 0.0:
        return f"{gs:.6f}"
    return f"{gs:.4e}"


def _annotate_bz0(ax, bz0_T: float, x0_label: str) -> None:
    """标出 0 处 Bz 实际值（Gs，高斯）。"""
    txt = f"{x0_label}：$B_z$ = {_format_bz_gs(bz0_T)} Gs\n（定义为 100%）"
    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.88},
    )


def plot_axis_field(data_dir: Path, out_path: Path) -> None:
    d = _load_csv(data_dir / "B_axis_z.csv")
    z_mm = d[:, 0] * 1e3
    bx, by, bz = d[:, 1], d[:, 2], d[:, 3]

    iz = _index_at_zero(d[:, 0])
    bz0 = float(bz[iz])
    ref, ref_note = _bz_percent_reference(bz, iz)
    pct_bz = 100.0 * bz / ref
    pct_bx = 100.0 * bx / ref
    pct_by = 100.0 * by / ref

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(8, 6.2),
        dpi=150,
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08},
    )
    ax_top.plot(z_mm, pct_bz, color="C0", linewidth=1.8, label=r"$B_z$ 相对值 (%)")
    ax_top.axhline(100.0, color="gray", linewidth=0.5, linestyle="--", alpha=0.55)
    ax_top.axvline(0.0, color="gray", linewidth=0.5, linestyle="-", alpha=0.45)
    ax_top.set_ylabel(r"相对 $B_z(z=0)$ (%)")
    title = "中心轴磁场相对分布（以 $z=0$ 处 $B_z$ 为 100%）"
    if ref_note:
        title += "\n" + ref_note
    ax_top.set_title(title)
    ax_top.grid(True, linestyle="--", alpha=0.35)
    ax_top.legend(loc="best", framealpha=0.92)
    _annotate_bz0(ax_top, bz0, "$z=0$")

    ax_bot.plot(z_mm, pct_bx, color="C1", linewidth=1.2, linestyle="--", label=r"$B_x$ / $|B_z(0)|$ (%)")
    ax_bot.plot(z_mm, pct_by, color="C2", linewidth=1.2, linestyle=":", label=r"$B_y$ / $|B_z(0)|$ (%)")
    ax_bot.axhline(0.0, color="gray", linewidth=0.5, linestyle="-", alpha=0.45)
    ax_bot.set_xlabel("轴向位置 $z$ (mm)")
    ax_bot.set_ylabel(r"$B_x,\,B_y$ 相对值 (%)")
    ax_bot.grid(True, linestyle="--", alpha=0.35)
    ax_bot.legend(loc="best", framealpha=0.92)

    fig.savefig(out_path, format="jpeg", pil_kwargs={"quality": 92})
    plt.close(fig)


def plot_radial_z0(data_dir: Path, out_path: Path) -> None:
    d = _load_csv(data_dir / "B_radial_line_z0.csv")
    r_mm = d[:, 0] * 1e3
    bx, by, bz = d[:, 1], d[:, 2], d[:, 3]
    brho = d[:, 4]

    ir = _index_at_zero(d[:, 0])
    bz0 = float(bz[ir])
    ref, ref_note = _bz_percent_reference(bz, ir)
    pct_bz = 100.0 * bz / ref
    pct_brho = 100.0 * brho / ref

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150, layout="constrained")
    ax.plot(r_mm, pct_bz, color="C0", linewidth=1.8, label=r"$B_z$ 相对值 (%)")
    ax.plot(r_mm, pct_brho, color="C3", linewidth=1.2, linestyle="-", label=r"$B_\rho$ / $|B_z(0)|$ (%)")
    ax.axhline(100.0, color="C0", linewidth=0.5, linestyle="--", alpha=0.45)
    ax.axvline(0.0, color="gray", linewidth=0.5, linestyle="-", alpha=0.45)
    ax.set_xlabel("径向距离 $\\rho$ (mm)，观测点 $(\\rho,0,0)$，$z=0$")
    ax.set_ylabel("相对值 (%)")
    title = (
        "平面 $z=0$ 内沿 $+x$ 轴：$B_z$、$B_\\rho$ 相对 $B_z(\\rho=0)$（100%）\n"
        r"平面电流时 $B_\rho\equiv 0$"
    )
    if ref_note:
        title += "\n" + ref_note
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", framealpha=0.92)
    _annotate_bz0(ax, bz0, r"$\rho=0$")

    fig.savefig(out_path, format="jpeg", pil_kwargs={"quality": 92})
    plt.close(fig)


def plot_radial_z_probe(data_dir: Path, out_path: Path) -> None:
    d = _load_csv(data_dir / "B_radial_line_z_probe.csv")
    r_mm = d[:, 0] * 1e3
    bx, by, bz = d[:, 1], d[:, 2], d[:, 3]
    brho = d[:, 4]

    ir = _index_at_zero(d[:, 0])
    bz0 = float(bz[ir])
    ref, ref_note = _bz_percent_reference(bz, ir)
    pct_bz = 100.0 * bz / ref
    pct_brho = 100.0 * brho / ref

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150, layout="constrained")
    (l1,) = ax1.plot(r_mm, pct_bz, color="C0", linewidth=1.8, label=r"$B_z$ 相对值 (%)")
    ax1.set_xlabel("径向距离 $\\rho$ (mm)，观测点 $(\\rho,0,z_{probe})$")
    ax1.set_ylabel(r"相对 $B_z(\rho=0)$ (%)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.axhline(100.0, color="C0", linewidth=0.5, linestyle="--", alpha=0.45)
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    (l2,) = ax2.plot(r_mm, pct_brho, color="C3", linewidth=1.5, linestyle="-", label=r"$B_\rho\,/\,|B_z(0)|$ (%)")
    ax2.set_ylabel(r"$B_\rho$ 相对 $|B_z(\rho=0)|$ (%)", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    z_probe_m = 0.5 * 1e-6 + 0.5e-6
    title = (
        f"略高于导体平面（$z_{{probe}} \\approx {z_probe_m*1e6:.3f}$ μm）\n"
        r"以 $\rho=0$ 处 $B_z$ 为 100%，$B_z$ 与 $B_\rho$ 均相对 $|B_z(0)|$ 表示"
    )
    if ref_note:
        title += "\n" + ref_note
    ax1.set_title(title)
    lines = [l1, l2]
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper right", framealpha=0.92)
    _annotate_bz0(ax1, bz0, r"$\rho=0$")

    fig.savefig(out_path, format="jpeg", pil_kwargs={"quality": 92})
    plt.close(fig)


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    plot_axis_field(data_dir, data_dir / "fig_B_axis_z.jpg")
    plot_radial_z0(data_dir, data_dir / "fig_B_radial_line_z0.jpg")
    plot_radial_z_probe(data_dir, data_dir / "fig_B_radial_line_z_probe.jpg")
    print("已保存:")
    for name in ("fig_B_axis_z.jpg", "fig_B_radial_line_z0.jpg", "fig_B_radial_line_z_probe.jpg"):
        print(f"  {data_dir / name}")


if __name__ == "__main__":
    main()
