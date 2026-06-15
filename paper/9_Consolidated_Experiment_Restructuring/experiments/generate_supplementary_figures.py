import json
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
COLOR_GEMMA = '#0891b2'

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
FIGURES = BASE / 'figures'
FIGURES.mkdir(exist_ok=True)


# ─── Figure 7: KV-Cache Recall Stability (C5 Resolution) ─────────────────────

def figure_7():
    fig, ax1 = plt.subplots()

    rounds = [4, 8, 12, 16, 20, 24, 28, 32]

    qwen4b_short = [0.90] * 8
    qwen4b_long = [0.933] * 8
    gemma_short = [1.00] * 8
    gemma_long = [1.00] * 8

    cache_tokens_4b = [879, 1721, 2568, 3414, 4260, 5108, 5954, 6800]

    ax1.plot(rounds, qwen4b_short, color=COLOR_SIG, marker='o', markersize=7,
             linewidth=2.2, label='Qwen-4B Short-Term (0.90)', zorder=5)
    ax1.plot(rounds, qwen4b_long, color=COLOR_APPLOOP, marker='s', markersize=7,
             linewidth=2.2, label='Qwen-4B Long-Term (0.933)', zorder=5)
    ax1.plot(rounds, gemma_short, color=COLOR_GEMMA, marker='^', markersize=7,
             linewidth=2.2, label='Gemma-4-E2B Short-Term (1.00)', zorder=5)
    ax1.plot(rounds, gemma_long, color=COLOR_COMPSIG, marker='D', markersize=7,
             linewidth=2.2, label='Gemma-4-E2B Long-Term (1.00)', zorder=5)

    ax1.axhline(y=1.0, color='#999999', linestyle='--', linewidth=1.5, alpha=0.7, label='Perfect Recall (1.0)')

    ax1.set_xlabel('Injection Round')
    ax1.set_ylabel('Recall Score')
    ax1.set_ylim(0.85, 1.05)
    ax1.set_xticks(rounds)

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(rounds)
    ax2.set_xticklabels([f'{t}' for t in cache_tokens_4b], fontsize=10)
    ax2.set_xlabel('Cache Token Count (Qwen-4B)', fontsize=12, color='#555555')
    ax2.tick_params(axis='x', colors='#555555')

    ax1.annotate('No degradation observed\nacross 32 injection rounds',
                 xy=(18, 0.875), fontsize=11, fontweight='bold', color='#16a34a',
                 ha='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0fdf4', edgecolor='#16a34a', alpha=0.9))

    ax1.annotate('Utilization gap,\nnot information loss',
                 xy=(28, 0.91), xytext=(26, 0.87),
                 fontsize=10, fontweight='bold', color='#b45309',
                 arrowprops=dict(arrowstyle='->', color='#b45309', lw=1.3),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff7ed', edgecolor='#b45309', alpha=0.9))

    ax1.set_title('KV-Cache Recall Stability Across Injection Rounds')
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig7_kvcache_recall_stability.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  [OK] fig7_kvcache_recall_stability.png')


# ─── Figure 8: DiskKVCache Break-Even Analysis (C6 Resolution) ────────────────

def figure_8():
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    sessions = np.arange(1, 21)

    cold_4b = 13.2
    disk_save_4b = 150.68
    disk_load_4b = 1.98
    breakeven_4b = 14

    cold_08b = 11.15
    disk_save_08b = 61.4
    disk_load_08b = 0.87
    breakeven_08b = 6

    cum_cold_4b = cold_4b * sessions
    cum_disk_4b = disk_save_4b + disk_load_4b * (sessions - 1)

    cum_cold_08b = cold_08b * sessions
    cum_disk_08b = disk_save_08b + disk_load_08b * (sessions - 1)

    ax_left.plot(sessions, cum_cold_4b, color=COLOR_SIG, marker='o', markersize=6,
                 linewidth=2.2, label=f'Cold Start ({cold_4b:.1f} ms/session)')
    ax_left.plot(sessions, cum_disk_4b, color=COLOR_APPLOOP, marker='s', markersize=6,
                 linewidth=2.2, label=f'Disk (save={disk_save_4b:.1f} + load={disk_load_4b:.2f}/session)')
    ax_left.axvline(x=breakeven_4b, color='#999999', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_left.annotate(f'Break-even\nN={breakeven_4b}',
                     xy=(breakeven_4b, cold_4b * breakeven_4b),
                     xytext=(breakeven_4b + 2.5, cold_4b * breakeven_4b - 30),
                     fontsize=11, fontweight='bold', color='#555555',
                     arrowprops=dict(arrowstyle='->', color='#555555', lw=1.3),
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', edgecolor='#555555', alpha=0.9))

    ax_left.set_xlabel('Number of Sessions')
    ax_left.set_ylabel('Cumulative Cost (ms)')
    ax_left.set_title('Qwen3.5-4B (58-token prefix)')
    ax_left.set_xticks(range(0, 21, 2))
    ax_left.legend(loc='upper left', framealpha=0.9, fontsize=10)

    ax_right.plot(sessions, cum_cold_08b, color=COLOR_SIG, marker='o', markersize=6,
                  linewidth=2.2, label=f'Cold Start ({cold_08b:.2f} ms/session)')
    ax_right.plot(sessions, cum_disk_08b, color=COLOR_APPLOOP, marker='s', markersize=6,
                  linewidth=2.2, label=f'Disk (save={disk_save_08b:.1f} + load={disk_load_08b:.2f}/session)')
    ax_right.axvline(x=breakeven_08b, color='#999999', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_right.annotate(f'Break-even\nN={breakeven_08b}',
                      xy=(breakeven_08b, cold_08b * breakeven_08b),
                      xytext=(breakeven_08b + 2.5, cold_08b * breakeven_08b - 15),
                      fontsize=11, fontweight='bold', color='#555555',
                      arrowprops=dict(arrowstyle='->', color='#555555', lw=1.3),
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', edgecolor='#555555', alpha=0.9))

    ax_right.set_xlabel('Number of Sessions')
    ax_right.set_ylabel('Cumulative Cost (ms)')
    ax_right.set_title('Qwen3.5-0.8B (58-token prefix)')
    ax_right.set_xticks(range(0, 21, 2))
    ax_right.legend(loc='upper left', framealpha=0.9, fontsize=10)

    fig.suptitle('DiskKVCache Break-Even Analysis', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig8_diskkvcache_breakeven.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  [OK] fig8_diskkvcache_breakeven.png')


# ─── Figure 9: SECM-H Paradigm Effect (C4 Resolution) ─────────────────────────

def figure_9():
    fig, ax = plt.subplots()

    conditions = ['Pre-scripted', 'Agent-driven\n(noisy)', 'Agent-driven\n(clean selective)']
    delta_q = [-0.141, 0.101, 0.122]
    colors = [COLOR_APPLOOP_PC, COLOR_SIG, COLOR_COMPSIG]

    bars = ax.bar(conditions, delta_q, color=colors, width=0.5,
                  edgecolor='#333333', linewidth=0.8)

    for bar, v in zip(bars, delta_q):
        y_off = 0.008 if v >= 0 else -0.015
        va = 'bottom' if v >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_off,
                f'{v:+.3f}', ha='center', va=va, fontsize=13, fontweight='bold')

    ax.axhline(y=0.0, color='#999999', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.annotate('',
                xy=(1, 0.101), xytext=(0, -0.141),
                arrowprops=dict(arrowstyle='<->', color=COLOR_SLIDING, lw=2.0))
    ax.annotate('+0.242 swing',
                xy=(0.5, -0.02), fontsize=12, fontweight='bold', color=COLOR_SLIDING,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f3ff', edgecolor=COLOR_SLIDING, alpha=0.9))

    ax.annotate('Paradigm bypasses\nmodule selection',
                xy=(0, -0.141), xytext=(-0.35, -0.08),
                fontsize=9, fontweight='bold', color=COLOR_APPLOOP_PC,
                arrowprops=dict(arrowstyle='->', color=COLOR_APPLOOP_PC, lw=1.0),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#fef2f2', edgecolor=COLOR_APPLOOP_PC, alpha=0.9))

    ax.annotate('Genuine module\nmanagement value',
                xy=(2, 0.122), xytext=(2.35, 0.06),
                fontsize=9, fontweight='bold', color=COLOR_COMPSIG,
                arrowprops=dict(arrowstyle='->', color=COLOR_COMPSIG, lw=1.0),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#f0fdf4', edgecolor=COLOR_COMPSIG, alpha=0.9))

    ax.set_ylabel('\u0394Q_content')
    ax.set_title('SECM-H Evaluation Paradigm Effect')
    ax.set_ylim(-0.2, 0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig9_secmh_paradigm_effect.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  [OK] fig9_secmh_paradigm_effect.png')


# ─── Figure 10: Contradiction Resolution Summary ──────────────────────────────

def figure_10():
    fig, ax = plt.subplots(figsize=(12, 7))

    contradictions = [
        ('C1: Speedup Consistency', 'RESOLVED',
         '2.55x across 6 experiments (\u00b10.5%)'),
        ('C2: Crossover Point', 'RESOLVED',
         'Crossover at ~0.7B parameters'),
        ('C3: Generation Causation', 'RESOLVED',
         'Prompt-format artifact (1.85x ratio)'),
        ('C4: SECM-H Paradigm', 'RESOLVED',
         '+0.242 swing from paradigm effect'),
        ('C5: Coverage Non-monotonicity', 'RESOLVED',
         'No degradation across 32 rounds'),
        ('C6: Prefix Caching', 'RESOLVED',
         'Break-even at 6-14 sessions'),
    ]

    labels = [c[0] for c in contradictions]
    statuses = [c[1] for c in contradictions]
    evidences = [c[2] for c in contradictions]

    y_pos = np.arange(len(labels))

    color_resolved = '#16a34a'
    color_partial = '#eab308'

    bar_colors = [color_resolved if s == 'RESOLVED' else color_partial for s in statuses]
    bar_widths = [1.0] * len(labels)

    bars = ax.barh(y_pos, bar_widths, color=bar_colors, height=0.55,
                   edgecolor='#333333', linewidth=0.8, alpha=0.85)

    for i, (bar, status, evidence) in enumerate(zip(bars, statuses, evidences)):
        ax.text(0.02, i, status, ha='left', va='center',
                fontsize=11, fontweight='bold', color='white')
        ax.text(1.04, i, evidence, ha='left', va='center',
                fontsize=10, color='#444444', style='italic')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlim(0, 2.8)
    ax.set_xticks([])
    ax.invert_yaxis()

    ax.set_title('Contradiction Resolution Status')
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig10_contradiction_resolution.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  [OK] fig10_contradiction_resolution.png')


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Generating supplementary figures for Paper 9 deep validation...')
    print(f'Output directory: {FIGURES}\n')

    figure_7()
    figure_8()
    figure_9()
    figure_10()

    print(f'\nAll 4 supplementary figures saved to: {FIGURES}')
