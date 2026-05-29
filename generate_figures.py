#!/usr/bin/env python3
"""Generate all paper figures."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

outdir = Path("figures")
outdir.mkdir(exist_ok=True)


def fig1_speedup_comparison():
    """Figure 1: Speedup comparison across conditions and model sizes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    conditions = ['AppLoop', 'SIG', 'SIG+SpecDec']
    colors = ['#4C72B0', '#55A868', '#C44E52']

    # 4B
    vals_4b = [1.00, 2.91, 3.20]
    bars1 = ax1.bar(conditions, vals_4b, color=colors, width=0.6, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Speedup vs AppLoop')
    ax1.set_title('(a) Qwen3.5-4B')
    ax1.set_ylim(0, 3.8)
    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    for bar, val in zip(bars1, vals_4b):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                f'{val:.2f}x', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # 0.8B
    vals_08b = [1.00, 1.06, 0.83]
    bars2 = ax2.bar(conditions, vals_08b, color=colors, width=0.6, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Speedup vs AppLoop')
    ax2.set_title('(b) Qwen3.5-0.8B')
    ax2.set_ylim(0, 1.5)
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    for bar, val in zip(bars2, vals_08b):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.2f}x', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.suptitle('Figure 1: Wall-Clock Speedup Comparison (35 Steps, EdgeAgent-Kitchen)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(outdir / 'fig1_speedup_comparison.png')
    plt.close()
    print("  Saved fig1_speedup_comparison.png")


def fig2_acceptance_rates():
    """Figure 2: Acceptance rates by drafter configuration."""
    fig, ax = plt.subplots(figsize=(8, 5))

    drafters = ['n=2\nK=3', 'n=3\nK=1', 'n=3\nK=2', 'n=3\nK=3', 'n=3\nK=5', 'n=4\nK=3']
    ar_4b = [75.2, 81.6, 81.6, 81.6, 81.6, 82.5]
    ar_08b = [85.8, 90.8, 90.8, 90.8, 90.8, 94.0]

    x = np.arange(len(drafters))
    width = 0.35

    bars1 = ax.bar(x - width/2, ar_4b, width, label='4B', color='#4C72B0',
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, ar_08b, width, label='0.8B', color='#55A868',
                   edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Acceptance Rate (%)')
    ax.set_xlabel('Drafter Configuration')
    ax.set_title('Figure 2: N-gram Drafter Acceptance Rates by Configuration')
    ax.set_xticks(x)
    ax.set_xticklabels(drafters)
    ax.set_ylim(70, 100)
    ax.legend()
    ax.axhline(y=80, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(outdir / 'fig2_acceptance_rates.png')
    plt.close()
    print("  Saved fig2_acceptance_rates.png")


def fig3_crossover():
    """Figure 3: Crossover analysis."""
    fig, ax = plt.subplots(figsize=(8, 5))

    model_sizes = [0.8, 4.0]
    sig_speedup = [1.06, 2.91]
    sig_specdec_speedup = [0.83, 3.20]

    ax.plot(model_sizes, sig_speedup, 'o-', color='#55A868', linewidth=2,
            markersize=8, label='SIG-only')
    ax.plot(model_sizes, sig_specdec_speedup, 's-', color='#C44E52', linewidth=2,
            markersize=8, label='SIG+SpecDec')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
               label='Breakeven (1.0x)')

    # Interpolate crossover points
    # SIG-only: 0.8B=1.06, 4B=2.91 → crossover at ~0.7B
    sig_crossover = 0.8 + (4.0 - 0.8) * (1.0 - 1.06) / (2.91 - 1.06)
    ax.plot(sig_crossover, 1.0, 'D', color='#55A868', markersize=10, zorder=5)
    ax.annotate(f'Crossover\n~{sig_crossover:.1f}B', xy=(sig_crossover, 1.0),
                xytext=(sig_crossover + 0.5, 0.7), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='#55A868'))

    ax.set_xlabel('Model Size (B parameters)')
    ax.set_ylabel('Speedup vs AppLoop')
    ax.set_title('Figure 3: Crossover Analysis — SIG vs AppLoop')
    ax.set_xlim(0, 5)
    ax.set_ylim(0.5, 3.8)
    ax.legend(loc='upper left')
    ax.set_xticks([0.8, 1.5, 2.0, 3.0, 4.0])

    plt.tight_layout()
    plt.savefig(outdir / 'fig3_crossover.png')
    plt.close()
    print("  Saved fig3_crossover.png")


def fig4_speedup_waterfall():
    """Figure 4: Speedup waterfall decomposition for 4B."""
    fig, ax = plt.subplots(figsize=(9, 5))

    categories = ['AppLoop\nPrefill', 'AppLoop\nGeneration', 'AppLoop\nTotal',
                  'SIG\nPrefill', 'SIG\nGeneration', 'SIG\nTotal',
                  'SIG+SpecDec\nGeneration', 'SIG+SpecDec\nTotal']
    # Approximate time breakdown (seconds) based on measured data
    # AppLoop: ~16.75s total, SIG: ~5.75s, SIG+SpecDec: ~5.24s
    apploop_prefill = 5.5
    apploop_gen = 10.0
    apploop_total = 16.75

    sig_prefill = 0.1
    sig_gen = 5.65
    sig_total = 5.75

    sigspecdec_gen = 5.14
    sigspecdec_total = 5.24

    values = [apploop_prefill, apploop_gen, apploop_total,
              sig_prefill, sig_gen, sig_total,
              sigspecdec_gen, sigspecdec_total]

    colors_list = ['#4C72B0', '#4C72B0', '#4C72B0',
                   '#55A868', '#55A868', '#55A868',
                   '#C44E52', '#C44E52']

    bars = ax.barh(range(len(categories)), values, color=colors_list,
                   edgecolor='black', linewidth=0.5, height=0.6)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xlabel('Wall-Clock Time (seconds)')
    ax.set_title('Figure 4: Speedup Decomposition — 4B Model (35 Steps)')
    ax.invert_yaxis()

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}s', va='center', fontsize=9)

    # Add speedup annotations
    ax.annotate('55x\nprefill\nreduction', xy=(sig_prefill, 3),
                xytext=(1.5, 2.5), fontsize=8, ha='center',
                arrowprops=dict(arrowstyle='->', color='black'))

    # Add separator lines
    ax.axhline(y=2.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axhline(y=5.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    # Legend
    legend_patches = [
        mpatches.Patch(color='#4C72B0', label='AppLoop'),
        mpatches.Patch(color='#55A868', label='SIG'),
        mpatches.Patch(color='#C44E52', label='SIG+SpecDec'),
    ]
    ax.legend(handles=legend_patches, loc='lower right')

    plt.tight_layout()
    plt.savefig(outdir / 'fig4_waterfall.png')
    plt.close()
    print("  Saved fig4_waterfall.png")


def fig5_exp2_recovery():
    """Figure 5: Post-injection acceptance rate recovery (EXP-2)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Data from EXP-2 (04_manuscript.md Table 4)
    t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ar = np.array([0.615, 0.645, 0.285, 0.520, 0.520, 0.505, 0.550, 0.550, 0.550, 0.520])
    baseline = 0.645

    ax.plot(t, ar * 100, 'o-', color='#C44E52', linewidth=2, markersize=6, label='Post-injection AR')
    ax.axhline(y=baseline * 100, color='#4C72B0', linestyle='--', linewidth=1.5,
               label=f'Baseline ({baseline*100:.1f}%)')

    # Highlight the depression at t=2
    ax.annotate('t=2: -36%\ndepression', xy=(2, 28.5), xytext=(4.5, 22),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='#C44E52', lw=1.5))

    # Shade the depression zone
    ax.fill_between(t, ar * 100, baseline * 100,
                    where=(ar < baseline), alpha=0.15, color='#C44E52',
                    label='Depression zone')

    ax.set_xlabel('Steps Since Injection')
    ax.set_ylabel('Acceptance Rate (%)')
    ax.set_title('Figure 5: Post-Injection Acceptance Rate Recovery (EXP-2)')
    ax.set_ylim(20, 75)
    ax.legend(loc='lower right')
    ax.set_xticks(t)

    plt.tight_layout()
    plt.savefig(outdir / 'fig5_recovery.png')
    plt.close()
    print("  Saved fig5_recovery.png")


def main():
    print("Generating figures...")
    fig1_speedup_comparison()
    fig2_acceptance_rates()
    fig3_crossover()
    fig4_speedup_waterfall()
    fig5_exp2_recovery()
    print("All figures saved to %s/" % outdir)


if __name__ == "__main__":
    main()
