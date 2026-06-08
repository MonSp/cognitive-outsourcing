#!/usr/bin/env python3
"""Generate publication-quality figures for the KV-Cache persistence paper (Section 7)."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

OUT_DIR = Path("paper/7_DiskKVCache/figures")

COLORS = {
    'cold':       '#C44E52',
    'in_memory':  '#4C72B0',
    'disk_fast':  '#55A868',
    'disk_comp':  '#CCB974',
}

STRATEGY_LABELS = [
    'Cold Start',
    'In-Memory PC',
    'DiskKVCache\n(fast)',
    'DiskKVCache\n(compressed)',
]


ALL_CONFIGS = [
    {
        'label': '0.8B\nCPU',
        'cold_ms': 84.59, 'in_memory_ms': 7.14,
        'disk_load_ms': 9.81, 'fallback_ms': 9.86,
        'save_comp_ms': 2.00, 'load_comp_ms': 75.49,
        'uncompressed_mb': 19.95, 'compressed_bytes': 342,
    },
    {
        'label': '0.8B\nGPU',
        'cold_ms': 14.75, 'in_memory_ms': 0.35,
        'disk_load_ms': 1.01, 'fallback_ms': 0.82,
        'save_comp_ms': 1.26, 'load_comp_ms': 4.71,
        'uncompressed_mb': 19.95, 'compressed_bytes': 342,
    },
    {
        'label': '4B\nCPU',
        'cold_ms': 322.43, 'in_memory_ms': 19.39,
        'disk_load_ms': 34.86, 'fallback_ms': 35.31,
        'save_comp_ms': 1.50, 'load_comp_ms': 288.29,
        'uncompressed_mb': 52.06, 'compressed_bytes': 345,
    },
    {
        'label': '4B\nGPU',
        'cold_ms': 12.48, 'in_memory_ms': 0.90,
        'disk_load_ms': 1.95, 'fallback_ms': 1.96,
        'save_comp_ms': 2.50, 'load_comp_ms': 16.95,
        'uncompressed_mb': 52.06, 'compressed_bytes': 345,
    },
]


def fig2_save_load_comparison(configs):
    n_configs = len(configs)
    n_strategies = 4

    fig, ax = plt.subplots(figsize=(10, 5))

    group_width = 0.80
    bar_width = group_width / n_strategies
    gap = 0.30
    group_centers = np.arange(n_configs) * (1 + gap)

    colors = [COLORS['cold'], COLORS['in_memory'], COLORS['disk_fast'], COLORS['disk_comp']]

    for si in range(n_strategies):
        x = group_centers + (si - (n_strategies - 1) / 2) * bar_width
        vals = []
        for cfg in configs:
            if si == 0:
                vals.append(cfg['cold_ms'])
            elif si == 1:
                vals.append(cfg['in_memory_ms'])
            elif si == 2:
                vals.append(cfg['disk_load_ms'])
            else:
                vals.append(cfg['load_comp_ms'])
        bars = ax.bar(x, vals, bar_width * 0.92, label=STRATEGY_LABELS[si],
                       color=colors[si], edgecolor='black', linewidth=0.5, zorder=3)

        for bi, (bar, v) in enumerate(zip(bars, vals)):
            cold = configs[bi]['cold_ms']
            speedup = cold / v
            if speedup >= 1.0:
                ann = f'{speedup:.1f}x'
            else:
                ann = f'{1/speedup:.1f}x'
            y_offset = v * 1.22 if v * 1.22 > v + 0.5 else v + 0.5
            ax.text(bar.get_x() + bar.get_width() / 2, y_offset, ann,
                    ha='center', va='bottom', fontsize=7, fontweight='bold',
                    color=colors[si])

    ax.set_yscale('log')
    ax.set_ylabel('Latency (ms)')
    ax.set_xlabel('Model Configuration')
    ax.set_xticks(group_centers)
    ax.set_xticklabels([cfg['label'] for cfg in configs])
    ax.set_title('Figure 2: Session Restoration Latency Comparison')
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='gray')
    ax.set_ylim(bottom=0.15, top=800)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig2_save_load_comparison.png')
    plt.close()
    print("  Saved fig2_save_load_comparison.png")


def fig3_compression_impact(configs):
    n_configs = len(configs)

    fig, (ax_size, ax_time) = plt.subplots(1, 2, figsize=(12, 4.5))

    x = np.arange(n_configs)
    bar_w = 0.32

    sizes_uncomp = [cfg['uncompressed_mb'] for cfg in configs]
    sizes_comp = [cfg['compressed_bytes'] / 1024 for cfg in configs]

    bars_u = ax_size.bar(x - bar_w / 2, sizes_uncomp, bar_w, label='Uncompressed',
                          color='#4C72B0', edgecolor='black', linewidth=0.5, zorder=3)
    bars_c = ax_size.bar(x + bar_w / 2, sizes_comp, bar_w, label='Compressed',
                          color='#CCB974', edgecolor='black', linewidth=0.5, zorder=3)

    for bi, (bu, bc, cfg) in enumerate(zip(bars_u, bars_c, configs)):
        ratio = cfg['uncompressed_mb'] * 1024 * 1024 / cfg['compressed_bytes']
        ax_size.annotate(f'{ratio:,.0f}x',
                         xy=(bc.get_x() + bc.get_width() / 2, bc.get_height()),
                         xytext=(bc.get_x() + bc.get_width() / 2, bc.get_height() * 5),
                         fontsize=7, ha='center', fontweight='bold',
                         arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
        ax_size.text(bu.get_x() + bu.get_width() / 2, bu.get_height() * 1.15,
                     f'{cfg["uncompressed_mb"]:.1f} MB', ha='center', va='bottom',
                     fontsize=7, fontweight='bold')
        ax_size.text(bc.get_x() + bc.get_width() / 2, bc.get_height() * 1.5,
                     f'{cfg["compressed_bytes"]} B', ha='center', va='bottom',
                     fontsize=7, fontweight='bold')

    ax_size.set_yscale('log')
    ax_size.set_ylabel('Size')
    ax_size.set_xlabel('Model Configuration')
    ax_size.set_xticks(x)
    ax_size.set_xticklabels([cfg['label'] for cfg in configs])
    ax_size.set_title('(a) Disk Footprint')
    ax_size.legend(loc='upper right', framealpha=0.9)
    ax_size.set_ylim(bottom=1e-2, top=200)

    bw = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bw
    group_colors = ['#4C72B0', '#CCB974', '#4C72B0', '#CCB974']
    group_labels = ['Save (fast)', 'Save (comp.)', 'Load (fast)', 'Load (comp.)']
    group_patterns = ['///', '///', '\\\\\\', '\\\\\\']

    fast_loads = [cfg['disk_load_ms'] for cfg in configs]
    comp_saves = [cfg['save_comp_ms'] for cfg in configs]
    comp_loads = [cfg['load_comp_ms'] for cfg in configs]
    cold_vals = [cfg['cold_ms'] for cfg in configs]
    data_groups = [cold_vals, comp_saves, fast_loads, comp_loads]

    for gi, (vals, color, label) in enumerate(zip(data_groups, group_colors, group_labels)):
        bx = x + offsets[gi]
        hatch = group_patterns[gi]
        ax_time.bar(bx, vals, bw * 0.88, label=label, color=color,
                     edgecolor='black', linewidth=0.5, hatch=hatch, zorder=3)

    ax_time.set_yscale('log')
    ax_time.set_ylabel('Latency (ms)')
    ax_time.set_xlabel('Model Configuration')
    ax_time.set_xticks(x)
    ax_time.set_xticklabels([cfg['label'] for cfg in configs])
    ax_time.set_title('(b) Compression Impact on Latency')
    ax_time.legend(loc='upper left', framealpha=0.9, ncol=2, fontsize=8)
    ax_time.set_ylim(bottom=0.5, top=1000)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig3_compression_impact.png')
    plt.close()
    print("  Saved fig3_compression_impact.png")


def fig4_breakeven_analysis(configs):
    N = np.arange(1, 21)
    panel_colors = ['#C44E52', '#4C72B0', '#55A868', '#CCB974']

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=False)
    axes = axes.flatten()

    for ax, cfg, pc in zip(axes, configs, panel_colors):
        T_cold = cfg['cold_ms']
        c_save_disk = cfg['disk_load_ms']
        c_load_disk = cfg['disk_load_ms']
        c_save_comp = cfg['save_comp_ms']
        c_load_comp = cfg['load_comp_ms']

        scenarios = [
            ('DiskKVCache (fast)', c_save_disk, c_load_disk, pc, '-'),
            ('DiskKVCache (compressed)', c_save_comp, c_load_comp, pc, '--'),
        ]

        cold_total = N * T_cold
        ax.plot(N, cold_total, 'k-o', linewidth=1.8, markersize=4,
                label='Cold Start', zorder=4)

        for sname, c_save, c_load, color, ls in scenarios:
            disk_total = c_save + (N - 1) * c_load
            ax.plot(N, disk_total, marker='s', markersize=4, linewidth=1.8,
                    linestyle=ls, color=color, label=sname, zorder=3)

            diff = cold_total - disk_total
            sign_change = np.where(np.diff(np.sign(diff)))[0]
            if len(sign_change) > 0:
                idx = sign_change[0]
                be_n = N[idx]
                be_val = cold_total[idx]
                ax.plot(be_n, be_val, '*', color=color, markersize=14, zorder=5,
                        markeredgecolor='black', markeredgewidth=0.5)
                ax.annotate(f'Break-even N={be_n}',
                            xy=(be_n, be_val),
                            xytext=(be_n + 2.5, be_val * 1.15),
                            fontsize=8, fontweight='bold', color=color,
                            arrowprops=dict(arrowstyle='->', color=color, lw=1.2))
            elif diff[0] > 0 and diff[-1] > 0:
                ax.annotate('Always wins',
                            xy=(1, c_save + c_load),
                            xytext=(5, c_save + c_load * 2),
                            fontsize=8, fontweight='bold', color=color,
                            arrowprops=dict(arrowstyle='->', color=color, lw=1))
            elif diff[0] < 0 and diff[-1] < 0:
                ax.annotate('Never wins',
                            xy=(10, c_save + 9 * c_load),
                            xytext=(14, c_save + 9 * c_load * 1.5),
                            fontsize=8, fontweight='bold', color=color,
                            arrowprops=dict(arrowstyle='->', color=color, lw=1))

        ax.set_xlabel('Number of Sessions (N)')
        ax.set_ylabel('Cumulative Latency (ms)')
        ax.set_title(cfg['label'].replace('\n', ' '))
        ax.set_xticks(N)
        ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
        ax.set_xlim(0.5, 20.5)

    fig.suptitle('Figure 4: Break-Even Analysis — Cold Start vs DiskKVCache',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig4_breakeven_analysis.png')
    plt.close()
    print("  Saved fig4_breakeven_analysis.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating KV-Cache persistence figures...")
    fig2_save_load_comparison(ALL_CONFIGS)
    fig3_compression_impact(ALL_CONFIGS)
    fig4_breakeven_analysis(ALL_CONFIGS)
    print(f"All figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
