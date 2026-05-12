"""
读取 archimedean_spiral_inductance.py --fields 生成的 CSV，绘制三张图并保存为 JPG。
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


def plot_axis_field(data_dir: Path, out_path: Path) -> None:
    d = _load_csv(data_dir / "B_axis_z.csv")
    z_mm = d[:, 0] * 1e3
    bx, by, bz = d[:, 1], d[:, 2], d[:, 3]

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(8, 6.2),
        dpi=150,
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08},
    )
    ax_top.plot(z_mm, bz * 1e6, color="C0", linewidth=1.8, label=r"$B_z$")
    ax_top.axhline(0.0, color="gray", linewidth=0.5, linestyle="-", alpha=0.45)
    ax_top.axvline(0.0, color="gray", linewidth=0.5, linestyle="-", alpha=0.45)
    ax_top.set_ylabel(r"$B_z$ (μT)")
    ax_top.set_title("线圈几何中心轴上的磁场分布（观测点 $(0,0,z)$，$I=1$ mA）")
    ax_top.grid(True, linestyle="--", alpha=0.35)
    ax_top.legend(loc="best", framealpha=0.92)

    ax_bot.plot(z_mm, bx * 1e9, color="C1", linewidth=1.2, linestyle="--", label=r"$B_x$ (nT)")
    ax_bot.plot(z_mm, by * 1e9, color="C2", linewidth=1.2, linestyle=":", label=r"$B_y$ (nT)")
    ax_bot.axhline(0.0, color="gray", linewidth=0.5, linestyle="-", alpha=0.45)
    ax_bot.set_xlabel("轴向位置 $z$ (mm)")
    ax_bot.set_ylabel(r"$B_x,\,B_y$ (nT)")
    ax_bot.grid(True, linestyle="--", alpha=0.35)
    ax_bot.legend(loc="best", framealpha=0.92)

    fig.savefig(out_path, format="jpeg", pil_kwargs={"quality": 92})
    plt.close(fig)


def plot_radial_z0(data_dir: Path, out_path: Path) -> None:
    d = _load_csv(data_dir / "B_radial_line_z0.csv")
    r_mm = d[:, 0] * 1e3
    bx, by, bz = d[:, 1], d[:, 2], d[:, 3]
    brho = d[:, 4]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150, layout="constrained")
    ax.plot(r_mm, bz * 1e6, color="C0", linewidth=1.8, label=r"$B_z$")
    ax.plot(r_mm, brho * 1e12, color="C3", linewidth=1.2, linestyle="-", label=r"$B_\rho=\sqrt{B_x^2+B_y^2}$ (pT)")
    ax.set_xlabel("径向距离 $r$ (mm)，观测点 $(r,0,0)$，$z=0$")
    ax.set_ylabel(r"$B_z$ (μT)；$B_\rho$ (pT)")
    ax.set_title(
        "线圈平面 $z=0$ 内沿 $x$ 轴的磁场\n"
        r"平面电流 $\mathbf{J}\parallel xy$ 时，严格 $z=0$ 上有 $B_x=B_y=0$，故 $B_\rho=0$"
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", framealpha=0.92)
    fig.savefig(out_path, format="jpeg", pil_kwargs={"quality": 92})
    plt.close(fig)


def plot_radial_z_probe(data_dir: Path, out_path: Path) -> None:
    d = _load_csv(data_dir / "B_radial_line_z_probe.csv")
    r_mm = d[:, 0] * 1e3
    bx, by, bz = d[:, 1], d[:, 2], d[:, 3]
    brho = d[:, 4]

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150, layout="constrained")
    (l1,) = ax1.plot(r_mm, bz * 1e6, color="C0", linewidth=1.8, label=r"$B_z$")
    ax1.set_xlabel("径向距离 $r$ (mm)，观测点 $(r,0,z_{probe})$")
    ax1.set_ylabel(r"$B_z$ (μT)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    (l2,) = ax2.plot(r_mm, brho * 1e6, color="C3", linewidth=1.5, linestyle="-", label=r"$B_\rho=\sqrt{B_x^2+B_y^2}$")
    ax2.set_ylabel(r"$B_\rho$ (μT)", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    z_probe_m = 0.5 * 1e-6 + 0.5e-6  # 与 run_field_study 一致：半高 + 0.5 μm
    ax1.set_title(
        f"略高于导体平面的径向扫描（$z_{{probe}} \\approx {z_probe_m*1e6:.3f}$ μm）\n"
        r"可显示非零的平面内横向分量 $B_\rho$，$I=1$ mA"
    )
    lines = [l1, l2]
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper right", framealpha=0.92)
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
