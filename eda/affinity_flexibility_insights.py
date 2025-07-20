import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


df = pd.read_csv('core_timetest.round0.csv')
df = df[df['pose_rank'] != 20] 


df['complex_id'] = df['Target'] + '_' + df['Molecule ID']

df['rank_type'] = df['pose_rank'].apply(lambda x: 'Top Rank (0)' if x == 0 else 'Other Ranks (1-19)')

success_df = df[df['pose_rank'] == 0].copy()
success_df['success'] = success_df['rmsd'].apply(lambda x: 1 if x < 2 else 0)
success_map = success_df.set_index('complex_id')['success'].to_dict()
df['success'] = df['complex_id'].map(success_map)

plt.figure(figsize=(10, 7), dpi=300)

violin_palette = {"Top Rank (0)": "#FDF5D8", "Other Ranks (1-19)": "#C8F0F2"}  

box_palette = {"Top Rank (0)": "#E6C878", "Other Ranks (1-19)": "#5FB2B5"}  

sns.violinplot(
    x='rank_type', y='rmsd', data=df,
    palette=violin_palette, 
    inner=None, 
    cut=0, 
    linewidth=1.5,  
    saturation=0.9,  
    order=["Top Rank (0)", "Other Ranks (1-19)"],
    alpha=1  
)


sns.boxplot(
    x='rank_type', y='rmsd', data=df,
    width=0.15, 
    fliersize=0,
    boxprops=dict(alpha=0.5, edgecolor='black', linewidth=1.5),  
    whiskerprops=dict(color='black', linewidth=1.5),
    medianprops=dict(color='gold', linewidth=2),
    palette=box_palette,  
    order=["Top Rank (0)", "Other Ranks (1-19)"]
)


plt.axhline(2, color='#7E3F8F', linestyle='--', alpha=0.8, linewidth=2)


ax = plt.gca()
medians = []
for i, category in enumerate(["Top Rank (0)", "Other Ranks (1-19)"]):
    median_val = df[df['rank_type'] == category]['rmsd'].median()
    medians.append(median_val)
    plt.text(i, median_val + 0.5, f'Median: {median_val:.2f}Å', 
             ha='center', fontsize=11, weight='bold',
             bbox=dict(facecolor='white', alpha=0.8))


plt.text(0.5, 2.2, 'Success Threshold (2Å)', 
         color='#7E3F8F', fontsize=10, ha='center')


plt.title('Top-Ranked Poses Exhibit Significantly Lower RMSD Values', fontsize=14)
plt.xlabel('Pose Rank Category', fontsize=12)
plt.ylabel('RMSD (Å)', fontsize=12)
plt.ylim(-0.5, df['rmsd'].quantile(0.99) + 1)
sns.despine()
plt.grid(axis='y', alpha=0.2)

plt.tight_layout()
plt.savefig('rmsd_distribution_comparison_optimized.png', bbox_inches='tight')

rank_rmsd = df.groupby('pose_rank')['rmsd'].median().reset_index()

plt.figure(figsize=(12, 5), dpi=300)

cmap = LinearSegmentedColormap.from_list('rd_gn', ["#0D8B43", "#F28147"])

for i in range(0, 20): 
    rmsd_val = rank_rmsd[rank_rmsd['pose_rank'] == i]['rmsd'].values[0]
    color = cmap(i/19)  
    plt.bar(i, rmsd_val, color=color, edgecolor='white', linewidth=1)
    
    if i == 0:
        plt.text(i, rmsd_val + 0.2, '★', 
                fontsize=20, ha='center', 
                color='gold', weight='bold')
    else:
        plt.text(i, rmsd_val/2, f'{rmsd_val:.1f}', 
                ha='center', va='center', 
                color='white', weight='bold')

plt.axhline(2, color='#7E3F8F', linestyle='--', alpha=0.8, linewidth=2)
plt.fill_between([-0.5, 19.5], 0, 2, color='#7E3F8F', alpha=0.08)

plt.title('Median RMSD Progressively Increases with Pose Rank', fontsize=14)
plt.xlabel('Pose Rank', fontsize=12)
plt.ylabel('Median RMSD (Å)', fontsize=12)
plt.xlim(-0.5, 19.5)
plt.ylim(0, rank_rmsd['rmsd'].max() * 1.1)
plt.xticks(range(0, 20, 1))  # 0-19
plt.grid(axis='y', alpha=0.2)
sns.despine()

plt.tight_layout()
plt.savefig('rmsd_by_rank_position.png', bbox_inches='tight')

sample_complexes = np.random.choice(df['complex_id'].unique(), 12, replace=False)
sample_df = df[df['complex_id'].isin(sample_complexes)]

sample_df.loc[:, 'success_label'] = sample_df['success'].map({
    0: 'Failed (RMSD≥2Å)',
    1: 'Success (RMSD<2Å)'
})

plt.figure(figsize=(14, 8), dpi=300)

palette = {"Top Rank (0)": "#E64B35", "Other Ranks (1-19)": "#4DBBD5"}

markers_dict = {
    "Success (RMSD<2Å)": "P",  
    "Failed (RMSD≥2Å)": "X"   
}

scatter = sns.scatterplot(
    x='complex_id', y='rmsd', 
    hue='rank_type', 
    style='success_label',
    data=sample_df,
    palette=palette,
    markers=markers_dict,  
    hue_order=["Top Rank (0)", "Other Ranks (1-19)"],
    style_order=["Success (RMSD<2Å)", "Failed (RMSD≥2Å)"],
    s=150, 
    alpha=0.9,
    edgecolor='w',
    linewidth=1.2
)

for complex_id in sample_complexes:
    complex_data = sample_df[sample_df['complex_id'] == complex_id]
    top_rank = complex_data[complex_data['pose_rank'] == 0]['rmsd'].values[0]
    other_mean = complex_data[complex_data['pose_rank'] > 0]['rmsd'].mean()
    
    plt.plot(
        [complex_id, complex_id], [top_rank, other_mean],
        color='gray', alpha=0.4, linewidth=1.5, zorder=1
    )
    
    diff_percent = (other_mean - top_rank) / max(top_rank, 0.1) * 100  
    plt.text(
        complex_id, (top_rank + other_mean)/2, 
        f'+{diff_percent:.0f}%', 
        ha='center', va='center', 
        fontsize=10, bbox=dict(facecolor='white', alpha=0.8),
        zorder=10 
    )


plt.axhline(2, color='#7E3F8F', linestyle='--', alpha=0.8, linewidth=2)
plt.text(
    0.01, 0.02, 'Success Threshold (RMSD < 2Å)',
    transform=plt.gca().transAxes,
    color='#7E3F8F', fontsize=11
)

plt.annotate(f'Random Sample: {", ".join(sample_complexes[:3])}...', 
             xy=(0.02, 0.98), xycoords='figure fraction',
             fontsize=9, alpha=0.7, ha='left')


from matplotlib.lines import Line2D


legend_elements = [

    Line2D([0], [0], 
           marker='o', 
           color='w', 
           label='Top Rank (0)',
           markerfacecolor='#E64B35', 
           markersize=12),
    Line2D([0], [0], 
           marker='o', 
           color='w', 
           label='Other Ranks (1-19)',
           markerfacecolor='#4DBBD5', 
           markersize=12),
    
    Line2D([0], [0], 
           marker='P', 
           color='w', 
           label='Success (RMSD<2Å)',
           markerfacecolor='grey', 
           markersize=12),
    Line2D([0], [0], 
           marker='X', 
           color='w', 
           label='Failed (RMSD≥2Å)',
           markerfacecolor='grey', 
           markersize=12)
]


legend = plt.legend(
    handles=legend_elements,
    title='Rank Type & Prediction Status',
    loc='upper right',
    frameon=True,
    framealpha=0.9,
    fontsize=10,
    title_fontsize=11,
    ncol=2  
)


plt.title('Top-Ranked Poses Outperform Lower Ranks in Selected Complexes', fontsize=16, pad=20)
plt.xlabel('Complex ID', fontsize=13, labelpad=10)
plt.ylabel('RMSD (Å)', fontsize=13, labelpad=10)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', alpha=0.2)
sns.despine()

plt.tight_layout()
plt.savefig('complex_comparison_scatter.png', bbox_inches='tight')