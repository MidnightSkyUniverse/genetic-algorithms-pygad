"""
ga_tsp_analyses.py - Statistical analysis and visualization of TSP experiment results

This script analyzes experiment results stored in experiment_results.csv and generates:
- Stability statistics (coefficient of variation, standard deviation)
- Comparative visualizations (boxplots, scatter plots, histograms)
- Recommendations for improving experiment reliability

Run this after main.py to analyze accumulated results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Visualization configuration
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
sns.set_palette("husl")

# Load experiment results
df = pd.read_csv('experiment_results.csv')

# Create output directory for analysis results
output_dir = Path('outputs/analysis')
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("STABILITY ANALYSIS - GENETIC ALGORITHM TSP EXPERIMENTS")
print("=" * 80)

# ============================================================================
# 1. BASIC STATISTICS FOR EACH EXPERIMENT
# ============================================================================

print("\n" + "=" * 80)
print("1. STABILITY OF EACH EXPERIMENT")
print("=" * 80)

# Calculate statistics grouped by experiment
stability_stats = df.groupby('experiment_name').agg({
    'best_distance': ['count', 'mean', 'std', 'min', 'max'],
    'generations_completed': ['mean', 'std']
}).round(2)

stability_stats.columns = ['_'.join(col).strip() for col in stability_stats.columns]

# Calculate coefficient of variation (CV) - key stability metric
# Lower CV = more stable/reproducible results
stability_stats['cv_distance'] = (stability_stats['best_distance_std'] /
                                  stability_stats['best_distance_mean'] * 100).round(2)

# Sort by coefficient of variation (most stable first)
stability_stats = stability_stats.sort_values('cv_distance')

print("\nCoefficient of Variation (CV) - lower values indicate more stable experiments:")
print(stability_stats[['best_distance_count', 'best_distance_mean', 'best_distance_std',
                       'best_distance_min', 'best_distance_max', 'cv_distance']])

# Save detailed statistics to CSV
stability_stats.to_csv(output_dir / 'stability_summary.csv')
print(f"\n✓ Detailed statistics saved to: {output_dir / 'stability_summary.csv'}")

# ============================================================================
# 2. PARENT SELECTION METHOD ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("2. COMPARISON OF PARENT SELECTION METHODS")
print("=" * 80)

# Group results by selection method
selection_stats = df.groupby('parent_selection_type').agg({
    'best_distance': ['count', 'mean', 'std', 'min', 'max']
}).round(2)

selection_stats.columns = ['_'.join(col).strip() for col in selection_stats.columns]
selection_stats['cv'] = (selection_stats['best_distance_std'] /
                         selection_stats['best_distance_mean'] * 100).round(2)

print("\nSelection methods (sorted by average quality):")
print(selection_stats.sort_values('best_distance_mean'))

# ============================================================================
# 3. VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 80)
print("3. GENERATING VISUALIZATIONS")
print("=" * 80)

# 3.1 BOXPLOT - Compare all experiments
fig, ax = plt.subplots(figsize=(16, 10))

# Sort experiments by median distance for better readability
exp_order = df.groupby('experiment_name')['best_distance'].median().sort_values().index

sns.boxplot(data=df, y='experiment_name', x='best_distance',
            order=exp_order, ax=ax)
ax.set_xlabel('Route Distance (best_distance)', fontsize=12, fontweight='bold')
ax.set_ylabel('Experiment', fontsize=12, fontweight='bold')
ax.set_title('Experiment Stability Comparison - Boxplot\n(sorted by median)',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add annotations showing number of trials
for i, exp in enumerate(exp_order):
    count = df[df['experiment_name'] == exp].shape[0]
    median = df[df['experiment_name'] == exp]['best_distance'].median()
    ax.text(median, i, f' n={count}', va='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig(output_dir / 'boxplot_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'boxplot_comparison.png'}")
plt.close()

# 3.2 SCATTER PLOT - Stability vs Quality tradeoff
fig, ax = plt.subplots(figsize=(12, 8))

# Calculate summary statistics for each experiment
exp_summary = df.groupby('experiment_name').agg({
    'best_distance': ['mean', 'std', 'count']
}).reset_index()
exp_summary.columns = ['experiment_name', 'mean_distance', 'std_distance', 'count']

# Create scatter plot - point size represents number of runs
scatter = ax.scatter(exp_summary['mean_distance'], exp_summary['std_distance'],
                     s=exp_summary['count'] * 50, alpha=0.6, c=range(len(exp_summary)),
                     cmap='viridis', edgecolors='black', linewidth=1.5)

# Add labels for each experiment
for idx, row in exp_summary.iterrows():
    ax.annotate(row['experiment_name'],
                (row['mean_distance'], row['std_distance']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax.set_xlabel('Average Route Distance (lower = better)', fontsize=12, fontweight='bold')
ax.set_ylabel('Standard Deviation (lower = more stable)', fontsize=12, fontweight='bold')
ax.set_title('Stability vs Quality Tradeoff\n(point size = number of runs)',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(alpha=0.3)

# Add reference lines showing overall averages
ax.axhline(exp_summary['std_distance'].mean(), color='red', linestyle='--',
           alpha=0.5, label=f'Avg std: {exp_summary["std_distance"].mean():.1f}')
ax.axvline(exp_summary['mean_distance'].mean(), color='blue', linestyle='--',
           alpha=0.5, label=f'Avg mean: {exp_summary["mean_distance"].mean():.1f}')
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / 'stability_vs_quality.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'stability_vs_quality.png'}")
plt.close()

# 3.3 HISTOGRAMS - Distribution of results for each experiment
experiments = df['experiment_name'].unique()
n_exp = len(experiments)
n_cols = 3
n_rows = (n_exp + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
axes = axes.flatten() if n_exp > 1 else [axes]

for idx, exp in enumerate(sorted(experiments)):
    exp_data = df[df['experiment_name'] == exp]['best_distance']

    # Plot histogram with statistical lines
    axes[idx].hist(exp_data, bins=min(20, len(exp_data)),
                   alpha=0.7, color='steelblue', edgecolor='black')
    axes[idx].axvline(exp_data.mean(), color='red', linestyle='--',
                      linewidth=2, label=f'Mean: {exp_data.mean():.1f}')
    axes[idx].axvline(exp_data.median(), color='green', linestyle='--',
                      linewidth=2, label=f'Median: {exp_data.median():.1f}')

    axes[idx].set_title(f'{exp}\n(n={len(exp_data)}, std={exp_data.std():.1f})',
                        fontweight='bold')
    axes[idx].set_xlabel('Route Distance')
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(alpha=0.3)

# Hide empty subplots
for idx in range(n_exp, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Distribution of Results for Each Experiment',
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(output_dir / 'histograms_all.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'histograms_all.png'}")
plt.close()


# ============================================================================
# 4. CONCLUSIONS AND RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 80)
print("4. CONCLUSIONS AND RECOMMENDATIONS")
print("=" * 80)

# Identify best configurations
best_mean = stability_stats.nsmallest(3, 'best_distance_mean')
most_stable = stability_stats.nsmallest(3, 'cv_distance')
least_stable = stability_stats.nlargest(3, 'cv_distance')

print("\n🏆 TOP 3 - BEST AVERAGE RESULTS (shortest routes):")
for idx, (exp_name, row) in enumerate(best_mean.iterrows(), 1):
    print(f"{idx}. {exp_name}: {row['best_distance_mean']:.2f} (±{row['best_distance_std']:.2f})")

print("\n✅ TOP 3 - MOST STABLE (lowest CV):")
for idx, (exp_name, row) in enumerate(most_stable.iterrows(), 1):
    print(f"{idx}. {exp_name}: CV={row['cv_distance']:.2f}% (mean={row['best_distance_mean']:.2f})")

print("\n⚠️  TOP 3 - LEAST STABLE (highest CV):")
for idx, (exp_name, row) in enumerate(least_stable.iterrows(), 1):
    print(f"{idx}. {exp_name}: CV={row['cv_distance']:.2f}% (mean={row['best_distance_mean']:.2f})")

# Diagnose instability issues
print("\n" + "=" * 80)
print("INSTABILITY DIAGNOSTICS")
print("=" * 80)

# Check for experiments with insufficient trials
low_samples = stability_stats[stability_stats['best_distance_count'] < 5]
if len(low_samples) > 0:
    print(f"\n⚠️  Experiments with insufficient trials (< 5):")
    for exp_name, row in low_samples.iterrows():
        print(f"   - {exp_name}: only {int(row['best_distance_count'])} trials")
else:
    print("\n✓ All experiments have sufficient number of trials")

# Check for high variance in generation counts (may indicate premature convergence)
gen_stats = df.groupby('experiment_name')['generations_completed'].agg(['mean', 'std', 'min', 'max'])
high_gen_variance = gen_stats[gen_stats['std'] > gen_stats['mean'] * 0.3]
if len(high_gen_variance) > 0:
    print(f"\n⚠️  Experiments with high variance in generation count:")
    for exp_name, row in high_gen_variance.iterrows():
        print(
            f"   - {exp_name}: {row['mean']:.0f} ±{row['std']:.0f} generations "
            f"(range: {int(row['min'])}-{int(row['max'])})")
else:
    print("\n✓ Generation count is stable across all experiments")

# General recommendations based on data
print("\n" + "=" * 80)
print("📋 RECOMMENDATIONS:")
print("=" * 80)

avg_cv = stability_stats['cv_distance'].mean()
if avg_cv > 10:
    print(f"\n1. Average CV = {avg_cv:.1f}% - results are HIGHLY UNSTABLE")
    print("   Consider:")
    print("   - Increase number of runs per experiment (minimum 30 repetitions)")
    print("   - Use fixed random seed for reproducibility")
    print("   - Extend algorithm runtime (more generations)")
elif avg_cv > 5:
    print(f"\n1. Average CV = {avg_cv:.1f}% - results are MODERATELY UNSTABLE")
    print("   Consider:")
    print("   - Increase repetitions to 20-30 runs")
    print("   - Verify that parameters are appropriate for the problem")
else:
    print(f"\n1. Average CV = {avg_cv:.1f}% - results are relatively STABLE")

print(f"\n2. To improve comparability:")
print("   - Ensure each experiment has the same number of runs")
print("   - Use the same TSP instance set for all experiments")
print("   - Record additional metrics (execution time, population diversity)")

print("\n" + "=" * 80)
print("✓ ANALYSIS COMPLETE")
print(f"✓ All results saved in: {output_dir.absolute()}")
print("=" * 80)