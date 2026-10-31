'''
Comparison plot for the ablation studies (report Section 3.7).
Reads the history_<RUN_NAME>.csv files saved by script_mlp.py and draws
one panel per experiment group, so each change is compared against the baseline.
Run from script/stage_2_script:   uv run python plot_comparison.py
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = '../../result/stage_2_result/'

# each panel: (title, [(run name, label in legend), ...])
# the baseline is always first, so it gets the same color in every panel
PANELS = [
    ('Learning rate (Adam)', [
        ('baseline',  'lr 0.001 (baseline)'),
        ('lr_0.01',   'lr 0.01'),
        ('lr_0.0001', 'lr 0.0001'),
    ]),
    ('Optimizer', [
        ('baseline',  'Adam, lr 0.001 (baseline)'),
        ('sgd_0.01',  'SGD, lr 0.01'),
        ('sgd_0.001', 'SGD, lr 0.001'),
    ]),
    ('Architecture (hidden layers)', [
        ('baseline',    '256-128 (baseline)'),
        ('arch_small',  '64'),
        ('arch_1layer', '256'),
        ('arch_deep',   '512-256-128'),
        ('arch_deeper', '512-256-128-64'),
    ]),
]

# fixed color order: slot 1 is always the baseline
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
MARKERS = ['o', 's', '^', 'D', 'v']   # shapes too, so lines differ without color (print/colorblind)

fig, axes = plt.subplots(1, len(PANELS), figsize=(15, 4.6), sharey=True)

for ax, (title, runs) in zip(axes, PANELS):
    for i, (run, label) in enumerate(runs):
        h = pd.read_csv(f'{RESULTS_DIR}history_{run}.csv')
        ax.plot(h['epoch'], h['loss'], color=COLORS[i], marker=MARKERS[i],
                markersize=4.5, linewidth=2 if i == 0 else 1.6, label=label)
    ax.set_title(title, fontsize=12, weight='bold')
    ax.set_xlabel('Epoch')
    ax.set_xticks(range(0, 21, 5))
    ax.set_yscale('log')   # losses range from ~2 down to ~0.003, so a log scale shows them all
    ax.grid(True, which='major', color='#e4e4e0', linewidth=0.8)
    ax.grid(True, which='minor', color='#f1f1ee', linewidth=0.5)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=9, frameon=False)

axes[0].set_ylabel('Training loss (log scale)')
fig.suptitle('Training loss vs. epoch for each ablation (20 epochs, same random seed)', fontsize=13)
fig.tight_layout()
out = RESULTS_DIR + 'comparison.png'
plt.savefig(out, dpi=200, facecolor='white')
print('saved', out)