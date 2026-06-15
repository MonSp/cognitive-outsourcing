import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#cccccc',
    'axes.facecolor': '#fafafa',
    'axes.edgecolor': '#333333',
    'text.color': '#222222',
    'axes.labelcolor': '#222222',
    'xtick.color': '#222222',
    'ytick.color': '#222222',
})

COLOR_SIG = '#2563eb'
COLOR_APPLOOP = '#ea580c'
COLOR_COMPSIG = '#16a34a'
COLOR_APPLOOP_PC = '#dc2626'
COLOR_SLIDING = '#7c3aed'

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
FIGURES = BASE / 'figures'
FIGURES.mkdir(exist_ok=True)


def load(name):
    with open(RESULTS / name, 'r') as f:
        return json.load(f)


exp1 = load('exp-1_unified_speedup_analysis.json')
exp2 = load('exp-2_model_size_sweep_analysis.json')
exp3 = load('exp-3_composition_matrix_analysis.json')
exp4 = load('exp-4_paradigm_comparison_analysis.json')
exp5 = load('exp-5_pareto_frontier_analysis.json')
exp6 = load('exp-6_generation_causation_analysis.json')
exp7 = load('exp-7_cross_architecture_analysis.json')
exp8 = load('exp-8_agent_driven_analysis.json')


def ci_err(mean, lo, hi):
    return (mean - lo, hi - mean)


# ─── Figure 1: SIG Speedup Comparison (EXP-1) ────────────────────────────────

def figure_1():
    fig, ax = plt.subplots()

    labels = ['AppLoop', 'SIG', 'AppLoop-PC', 'Sliding']
    keys = ['AppLoop', 'SIG', 'AppLoop-PC', 'AppLoop-Sliding']
    colors = [COLOR_APPLOOP, COLOR_SIG, COLOR_APPLOOP_PC, COLOR_SLIDING]

    means = [exp1['conditions'][k]['wall_clock_s_mean'] for k in keys]
    ci_lo = [exp1['conditions'][k]['wall_clock_s_ci95_lo'] for k in keys]
    ci_hi = [exp1['conditions'][k]['wall_clock_s_ci95_hi'] for k in keys]
    err_lo = [m - lo for m, lo in zip(means, ci_lo)]
    err_hi = [hi - m for m, hi in zip(means, ci_hi)]

    bars = ax.bar(labels, means, color=colors, width=0.55, edgecolor='#333333',
                  linewidth=0.8, yerr=[err_lo, err_hi], capsize=5,
                  error_kw={'linewidth': 1.2, 'color': '#555555'})

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{m:.2f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

    su = exp1['speedup_analysis']
    ax.annotate(
        f'SIG Speedup: {su["speedup_mean"]:.2f}×\n(95% CI [{su["speedup_ci95_lo"]:.3f}, {su["speedup_ci95_hi"]:.3f}])',
        xy=(1, means[1]), xytext=(2.2, 55),
        fontsize=12, fontweight='bold', color=COLOR_SIG,
        arrowprops=dict(arrowstyle='->', color=COLOR_SIG, lw=1.5),
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#eff6ff', edgecolor=COLOR_SIG, alpha=0.9))

    ax.set_ylabel('Wall-Clock Time (s)')
    ax.set_title('SIG Speedup Under Unified Protocol (EXP-1)')
    ax.set_ylim(0, 85)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig1_sig_speedup_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  [OK] fig1_sig_speedup_comparison.png')


# ─── Figure 2: Speedup vs Model Size (EXP-2) ─────────────────────────────────

def figure_2():
    fig, ax = plt.subplots()

    sizes = ['0.8B', '4B']
    speedups = [exp2['speedup_by_size']['0.8B']['speedup_mean'],
                exp2['speedup_by_size']['4B']['speedup_mean']]
    ci_lo = [exp2['speedup_by_size']['0.8B']['speedup_ci95_lo'],
             exp2['speedup_by_size']['4B']['speedup_ci95_lo']]
    ci_hi = [exp2['speedup_by_size']['0.8B']['speedup_ci95_hi'],
             exp2['speedup_by_size']['4B']['speedup_ci95_hi']]
    err_lo = [m - lo for m, lo in zip(speedups, ci_lo)]
    err_hi = [hi - m for m, hi in zip(speedups, ci_hi)]

    bars = ax.bar(sizes, speedups, color=[COLOR_SIG, COLOR_SIG], width=0.45,
                  edgecolor='#333333', linewidth=0.8,
                  yerr=[err_lo, err_hi], capsize=6,
                  error_kw={'linewidth': 1.2, 'color': '#555555'})

    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{s:.2f}×', ha='center', va='bottom', fontsize=13, fontweight='bold')

    ax.axhline(y=1.0, color='#999999', linestyle='--', linewidth=1.5, label='Breakeven (1.0×)')
    ax.text(0.15, 1.03, 'Breakeven', fontsize=10, color='#666666', ha='left')

    ax.annotate('Crossover region\n~2B parameters',
                xy=(0.55, 1.65), fontsize=11, fontweight='bold', color='#b45309',
                ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff7ed', edgecolor='#b45309', alpha=0.9))
    ax.annotate('', xy=(0.5, 1.05), xytext=(0.5, 1.55),
                arrowprops=dict(arrowstyle='<->', color='#b45309', lw=1.5))

    ax.set_xlabel('Model Size (B params)')
    ax.set_ylabel('Speedup Factor')
    ax.set_title('SIG-vs-AppLoop Speedup vs Model Size (EXP-2)')
    ax.set_ylim(0, 3.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['0.8B', '4B'])
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig2_speedup_vs_model_size.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  [OK] fig2_speedup_vs_model_size.png')


# ─── Figure 3: Speedup vs Chain Depth (EXP-5) ────────────────────────────────

def figure_3():
    fig, ax = plt.subplots()

    depths = [5, 10, 20, 35, 50]

    def get_speedups(model_suffix):
        result = []
        ci_lo_list = []
        ci_hi_list = []
        for d in depths:
            key = f'D{d}_{model_suffix}_SIG'
            entry = exp5['depth_speedups'][key]
            result.append(entry['speedup_mean'])
            ci_lo_list.append(entry['speedup_ci95_lo'])
            ci_hi_list.append(entry['speedup_ci95_hi'])
        return result, ci_lo_list, ci_hi_list

    sp_4b, lo_4b, hi_4b = get_speedups('4B')
    sp_08b, lo_08b, hi_08b = get_speedups('0.8B')

    err_lo_4b = [m - lo for m, lo in zip(sp_4b, lo_4b)]
    err_hi_4b = [hi - m for m, hi in zip(sp_4b, hi_4b)]
    err_lo_08b = [m - lo for m, lo in zip(sp_08b, lo_08b)]
    err_hi_08b = [hi - m for m, hi in zip(sp_08b, hi_08b)]

    ax.errorbar(depths, sp_4b, yerr=[err_lo_4b, err_hi_4b],
                color=COLOR_SIG, marker='o', markersize=8, linewidth=2.2,
                capsize=5, capthick=1.5, label='4B (Qwen3.5-4B)', zorder=5)
    ax.errorbar(depths, sp_08b, yerr=[err_lo_08b, err_hi_08b],
                color=COLOR_APPLOOP, marker='s', markersize=8, linewidth=2.2,
                capsize=5, capthick=1.5, label='0.8B (Qwen3.5-0.8B)', zorder=5)

    ax.axhline(y=1.0, color='#999999', linestyle='--', linewidth=1.5, label='Breakeven (1.0×)')

    for d, s in zip(depths, sp_4b):
        ax.annotate(f'{s:.2f}×', (d, s), textcoords='offset points',
                    xytext=(0, 12), ha='center', fontsize=10, color=COLOR_SIG, fontweight='bold')

    for d, s in zip(depths, sp_08b):
        ax.annotate(f'{s:.2f}×', (d, s), textcoords='offset points',
                    xytext=(0, -18), ha='center', fontsize=10, color=COLOR_APPLOOP, fontweight='bold')

    ax.annotate('0.8B crosses\nbreakeven at D≈35',
                xy=(35, 1.10), xytext=(38, 0.55),
                fontsize=10, fontweight='bold', color='#b45309',
                arrowprops=dict(arrowstyle='->', color='#b45309', lw=1.3),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff7ed', edgecolor='#b45309', alpha=0.9))

    ax.set_xlabel('Chain Depth (steps)')
    ax.set_ylabel('Speedup Factor')
    ax.set_title('SIG Speedup vs Chain Depth (EXP-5)')
    ax.set_xticks(depths)
    ax.set_ylim(0, 4.5)
    ax.legend(loc='upper left', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig3_speedup_vs_chain_depth.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  [OK] fig3_speedup_vs_chain_depth.png')


# ─── Figure 4: Cross-Architecture Speedup (EXP-7) ────────────────────────────

def figure_4():
    fig, ax = plt.subplots()

    archs = ['Qwen 4B', 'Nemotron', 'Gemma']
    speedups = [
        exp7['speedups']['SIG_4B']['speedup_mean'],
        exp7['speedups']['SIG_nemotron']['speedup_mean'],
        exp7['speedups']['SIG_gemma']['speedup_mean'],
    ]
    ci_lo = [
        exp7['speedups']['SIG_4B']['speedup_ci95_lo'],
        exp7['speedups']['SIG_nemotron']['speedup_ci95_lo'],
        exp7['speedups']['SIG_gemma']['speedup_ci95_lo'],
    ]
    ci_hi = [
        exp7['speedups']['SIG_4B']['speedup_ci95_hi'],
        exp7['speedups']['SIG_nemotron']['speedup_ci95_hi'],
        exp7['speedups']['SIG_gemma']['speedup_ci95_hi'],
    ]
    err_lo = [m - lo for m, lo in zip(speedups, ci_lo)]
    err_hi = [hi - m for m, hi in zip(speedups, ci_hi)]
    colors = [COLOR_SIG, '#0891b2', '#7c3aed']

    bars = ax.bar(archs, speedups, color=colors, width=0.5,
                  edgecolor='#333333', linewidth=0.8,
                  yerr=[err_lo, err_hi], capsize=6,
                  error_kw={'linewidth': 1.2, 'color': '#555555'})

    for bar, s in zip(bars, speedups):
        y_off = 0.04 if s >= 1.0 else -0.12
        va = 'bottom' if s >= 1.0 else 'top'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_off,
                f'{s:.2f}×', ha='center', va=va, fontsize=13, fontweight='bold')

    ax.axhline(y=1.0, color='#999999', linestyle='--', linewidth=1.5, label='Baseline (1.0×)')

    for i, (arch, s) in enumerate(zip(archs, speedups)):
        verdict = 'Faster' if s > 1.0 else 'Slower'
        color = '#16a34a' if s > 1.0 else '#dc2626'
        ax.text(i, 0.15, verdict, ha='center', fontsize=10, color=color,
                fontweight='bold', style='italic')

    ax.set_ylabel('Speedup Factor')
    ax.set_title('Cross-Architecture SIG Speedup (EXP-7)')
    ax.set_ylim(0, 3.2)
    ax.legend(loc='upper right', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig4_cross_architecture_speedup.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  [OK] fig4_cross_architecture_speedup.png')


# ─── Figure 5: Speedup Decomposition (EXP-6) ─────────────────────────────────

def figure_5():
    fig, ax = plt.subplots()

    sig_gen = exp6['analysis']['decomposition']['SIG_standard']['gen_time_mean']
    sig_pre = exp6['analysis']['decomposition']['SIG_standard']['prefill_time_mean']
    app_gen = exp6['analysis']['decomposition']['AppLoop_standard']['gen_time_mean']
    app_pre = exp6['analysis']['decomposition']['AppLoop_standard']['prefill_time_mean']

    labels = ['SIG (4B)', 'AppLoop (4B)']
    prefill_vals = [sig_pre, app_pre]
    gen_vals = [sig_gen, app_gen]

    x = np.arange(len(labels))
    w = 0.45

    b1 = ax.bar(x, prefill_vals, w, label='Prefill Time', color='#60a5fa',
                edgecolor='#333333', linewidth=0.8)
    b2 = ax.bar(x, gen_vals, w, bottom=prefill_vals, label='Generation Time',
                color=COLOR_APPLOOP, edgecolor='#333333', linewidth=0.8)

    for i, (pre, gen) in enumerate(zip(prefill_vals, gen_vals)):
        total = pre + gen
        ax.text(i, total + 0.3, f'{total:.1f}s', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
        ax.text(i, pre / 2, f'{pre:.1f}s', ha='center', va='center',
                fontsize=10, color='white', fontweight='bold')
        ax.text(i, pre + gen / 2, f'{gen:.1f}s', ha='center', va='center',
                fontsize=10, color='white', fontweight='bold')

    ratio = exp6['analysis']['token_count_ratio']['ratio']
    prefill_ratio = app_pre / sig_pre

    ax.annotate(f'Prefill ratio: {prefill_ratio:.1f}×\n(SIG prefill nearly eliminated)',
                xy=(0, sig_pre), xytext=(0.55, 12),
                fontsize=11, fontweight='bold', color='#1e40af',
                arrowprops=dict(arrowstyle='->', color='#1e40af', lw=1.3),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#eff6ff', edgecolor='#1e40af', alpha=0.9))

    ax.annotate(f'Generation ratio: {ratio:.2f}×\n(SIG generates fewer tokens)',
                xy=(1, app_pre + app_gen / 2), xytext=(1.55, 22),
                fontsize=11, fontweight='bold', color='#9a3412',
                arrowprops=dict(arrowstyle='->', color='#9a3412', lw=1.3),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff7ed', edgecolor='#9a3412', alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Time (s)')
    ax.set_title('Speedup Decomposition: Prefill vs Generation (EXP-6)')
    ax.set_ylim(0, 24)
    ax.legend(loc='upper left', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig5_speedup_decomposition.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  [OK] fig5_speedup_decomposition.png')


# ─── Figure 6: Speedup Consistency Across Experiments ─────────────────────────

def figure_6():
    fig, ax = plt.subplots()

    exp_labels = ['EXP-1\n(Unified)', 'EXP-3\n(Composition)', 'EXP-4\n(Paradigm)',
                  'EXP-6\n(Decomposition)', 'EXP-8\n(Agent-Driven)']
    speedups = [
        exp1['speedup_analysis']['speedup_mean'],
        exp3['speedups']['C2_SIG_4B']['speedup_mean'],
        exp4['analysis']['paradigm_effects']['pre-scripted_clean']['speedup_mean'],
        exp6['conditions']['SIG_standard']['wall_clock_s_mean'] / exp6['conditions']['AppLoop_standard']['wall_clock_s_mean'],
        exp8['speedups']['4B_clean']['speedup_mean'],
    ]
    ci_lo = [
        exp1['speedup_analysis']['speedup_ci95_lo'],
        exp3['speedups']['C2_SIG_4B']['speedup_ci95_lo'],
        exp4['analysis']['paradigm_effects']['pre-scripted_clean']['speedup_ci95_lo'],
        speedups[3],
        exp8['speedups']['4B_clean']['speedup_ci95_lo'],
    ]
    ci_hi = [
        exp1['speedup_analysis']['speedup_ci95_hi'],
        exp3['speedups']['C2_SIG_4B']['speedup_ci95_hi'],
        exp4['analysis']['paradigm_effects']['pre-scripted_clean']['speedup_ci95_hi'],
        speedups[3],
        exp8['speedups']['4B_clean']['speedup_ci95_hi'],
    ]
    err_lo = [m - lo for m, lo in zip(speedups, ci_lo)]
    err_hi = [hi - m for m, hi in zip(speedups, ci_hi)]

    bars = ax.bar(exp_labels, speedups, color=COLOR_SIG, width=0.55,
                  edgecolor='#333333', linewidth=0.8,
                  yerr=[err_lo, err_hi], capsize=5,
                  error_kw={'linewidth': 1.2, 'color': '#555555'})

    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{s:.2f}×', ha='center', va='bottom', fontsize=12, fontweight='bold')

    mean_speedup = np.mean(speedups)
    ax.axhline(y=mean_speedup, color=COLOR_SIG, linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(len(exp_labels) - 0.5, mean_speedup + 0.03, f'Mean: {mean_speedup:.2f}×',
            fontsize=11, color=COLOR_SIG, fontweight='bold', ha='right')

    ax.axhline(y=1.0, color='#999999', linestyle='--', linewidth=1.0, alpha=0.5)

    ax.set_ylabel('SIG Speedup Factor (4B)')
    ax.set_title('SIG Speedup Consistency Across Experiments (4B Model)')
    ax.set_ylim(0, 3.2)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig6_speedup_consistency.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  [OK] fig6_speedup_consistency.png')


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Generating publication figures for Paper 9 CO-SIG experiments...')
    print(f'Output directory: {FIGURES}\n')

    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    figure_6()

    print(f'\nAll 6 figures saved to: {FIGURES}')
