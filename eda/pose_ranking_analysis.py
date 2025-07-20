import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.ticker as mtick

# Read data and remove pose_rank=20
df = pd.read_csv('benchmark/docking/timesplit/core_timetest.round0.csv')
df = df[df['pose_rank'] != 20]  # Remove data with pose_rank=20

# Filter pose_rank=0
core_df = df[df['pose_rank'] == 0].copy()

# Calculate prediction success flag
core_df['success'] = core_df['rmsd'].apply(lambda x: 1 if x < 2 else 0)
success_rate = core_df['success'].mean()

# ============ Affinity-Energy Scatter Plot ============
plt.figure(figsize=(12, 8), dpi=300)

# Calculate density for color mapping
xy = np.vstack([core_df['pIC50'], core_df['energy']])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = core_df['pIC50'].iloc[idx], core_df['energy'].iloc[idx], z[idx]

# Draw scatter plot
scatter = plt.scatter(
    x, y, c=z, 
    s=60,  # Increase point size
    cmap='viridis', 
    alpha=0.85,
    edgecolor='w',
    linewidth=0.8
)

# Add success region
plt.axhline(y=core_df[core_df['success']==1]['energy'].min(), 
            color='#E64B35', linestyle='--', alpha=0.6, linewidth=1.5)
plt.axvline(x=core_df[core_df['success']==1]['pIC50'].min(), 
            color='#E64B35', linestyle='--', alpha=0.6, linewidth=1.5)

# Add trend line
sns.regplot(
    data=core_df, x='pIC50', y='energy', 
    scatter=False, 
    color='#F39C12',
    line_kws={'lw':2.5, 'alpha':0.8}
)

# Add annotation
plt.annotate(f'Success Rate: {success_rate:.1%}\n(RMSD < 2Å)', 
             xy=(0.05, 0.95), xycoords='axes fraction',
             fontsize=12, bbox=dict(alpha=0.85))

# Beautify settings
cbar = plt.colorbar(scatter)
cbar.set_label('Point Density', fontsize=12)
plt.xlabel('Experimental pIC50 (Affinity)', fontsize=12)
plt.ylabel('Predicted Binding Energy', fontsize=12)
plt.title('Correlation Between Experimental Affinity and Predicted Binding Energy', fontsize=14)
plt.grid(alpha=0.2)
sns.despine()

plt.tight_layout()
plt.savefig('affinity_energy_scatter.png', bbox_inches='tight')

# ============ Flexibility-Success Rate Line Plot ============
# Create bins
bins = [0, 5, 10, 15, 100]
labels = ['Low (0-5)', 'Medium (6-10)', 'High (11-15)', 'Very High (>15)']
core_df['flexibility'] = pd.cut(core_df['num_torsions'], bins=bins, labels=labels)

# Calculate group statistics
grouped = core_df.groupby('flexibility').agg(
    success_rate=('success', 'mean'),
    avg_rmsd=('rmsd', 'mean'),
    count=('success', 'count')
).reset_index()

# Create dual-axis chart
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

# Success rate line
ax1.plot(
    grouped['flexibility'], grouped['success_rate'], 
    marker='o', markersize=10, 
    color='#3C5488', linewidth=3, 
    label='Success Rate'
)
ax1.set_ylabel('Success Rate (RMSD < 2Å)', fontsize=12)
ax1.set_ylim(0, 1.0)
ax1.grid(alpha=0.2)
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))

# Average RMSD bar chart
ax2 = ax1.twinx()
bars = ax2.bar(
    grouped['flexibility'], grouped['avg_rmsd'],
    alpha=0.3, color='#E64B35',
    width=0.6, label='Avg RMSD'
)
ax2.set_ylabel('Average RMSD (Å)', fontsize=12)
ax2.set_ylim(0, grouped['avg_rmsd'].max()*1.2)

# Add value labels to bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height*1.05,
            f'{height:.1f}', 
            ha='center', va='bottom', fontsize=10)

# Add count annotation
for i, row in enumerate(grouped.itertuples()):
    ax1.text(i, row.success_rate+0.05, f"n={row.count}", 
            ha='center', fontsize=10)

# Beautify settings
ax1.set_xlabel('Ligand Flexibility (Number of Torsions)', fontsize=12)
ax1.set_title('Prediction Success Decreases with Increasing Ligand Flexibility', fontsize=14)

# Merge legends
lines, labels = ax1.get_legend_handles_labels()
bars, bar_labels = ax2.get_legend_handles_labels()
ax2.legend(lines + bars, labels + bar_labels, 
          loc='upper center', ncol=2, frameon=True)

sns.despine()

plt.tight_layout()
plt.savefig('flexibility_success_line.png', bbox_inches='tight')