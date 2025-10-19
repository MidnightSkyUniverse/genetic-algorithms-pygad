"""
utils_unified_analyses.py - Helper functions for multi-run GA experiment analysis

This module provides functions for:
- Collecting and aggregating data from multiple experiment runs
- Computing stability metrics (CV, mean, std) across runs
- Generating comparative visualizations
- Creating summary reports

Key Metrics:
- CV (Coefficient of Variation): (std/mean) × 100% - measures relative variability
  Lower CV = more stable/reproducible results
- Compromise Score: Normalized balance between quality (low distance) and stability (low CV)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Color palette for consistent visualization style
COLORS = {
    'primary': '#5c26ff',      # Purple - primary metric
    'secondary': '#9c31ff',    # Light purple - secondary metric
    'accent': '#ffba0a',       # Yellow - highlights
    'extended_palette': [      # Extended palette for multiple categories
        '#5c26ff', '#ffba0a', '#ff6b6b', '#4ecdc4', '#95e1d3',
        '#c7a8ff', '#ffd93d', '#ff8fab', '#6bcf7f', '#a8daff',
        '#ffb347', '#b19cd9', '#77dd77', '#ff6f91', '#aec6cf'
    ]
}

# Matplotlib configuration for publication-quality plots
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 13

sns.set_palette(COLORS['extended_palette'])


# ============================================================================
# DATA COLLECTION FUNCTIONS
# ============================================================================

def collect_all_experiment_data(experiments_dir='experiments'):
    """
    Collect and aggregate data from all experiment runs.

    This function performs three main tasks:
    1. Finds best single result for each (run × experiment) combination
    2. Calculates stability statistics for each (run × experiment)
    3. Merges both datasets with parameter information

    Args:
        experiments_dir: Directory containing experiment_results_run_*.csv files

    Returns:
        tuple: (best_runs_df, stability_df, full_df) or (None, None, None) on error
            - best_runs_df: Best result per (run × experiment)
            - stability_df: Stability metrics per (run × experiment)
            - full_df: Merged data with all information
    """

    base_path = Path(experiments_dir)

    print("=" * 80)
    print("STEP 1: COLLECTING DATA FROM EXPERIMENTS")
    print("=" * 80)

    # Find all result files matching pattern
    result_files = sorted(base_path.glob('experiment_results_run_*.csv'))

    if not result_files:
        print(f"❌ No experiment_results_run_*.csv files found in {experiments_dir}/")
        return None, None, None

    print(f"\nFound {len(result_files)} result files\n")

    # Collect best individual results
    all_best_runs = []

    for file in result_files:
        run_number = file.stem.replace('experiment_results_', '')

        try:
            df = pd.read_csv(file)

            # Find best result for each experiment in this run
            best_per_experiment = df.loc[df.groupby('experiment_name')['best_distance'].idxmin()].copy()

            # Remove metadata columns (not needed for analysis)
            columns_to_drop = ['timestamp', 'run_id']
            best_per_experiment = best_per_experiment.drop(
                columns=[col for col in columns_to_drop if col in best_per_experiment.columns]
            )

            # Add run identifier
            best_per_experiment.insert(0, 'run', run_number)
            all_best_runs.append(best_per_experiment)

            print(f"✓ {run_number}: {len(best_per_experiment)} experiments")

        except Exception as e:
            print(f"✗ Error processing {file.name}: {e}")

    if not all_best_runs:
        print("❌ Failed to collect best results!")
        return None, None, None

    best_runs_df = pd.concat(all_best_runs, ignore_index=True)

    # Calculate stability statistics
    all_stability = []

    for file in result_files:
        run_number = file.stem.replace('experiment_results_', '')

        try:
            df = pd.read_csv(file)

            # Aggregate statistics per experiment
            stability = df.groupby('experiment_name').agg({
                'best_distance': ['count', 'mean', 'std', 'min', 'max'],
                'generations_completed': ['mean', 'std']
            }).round(2)

            # Flatten column names
            stability.columns = ['_'.join(col).strip() for col in stability.columns]

            # Calculate Coefficient of Variation (key stability metric)
            # CV = (std / mean) × 100%
            # Lower CV indicates more consistent/reproducible results
            stability['cv_distance'] = (
                    stability['best_distance_std'] / stability['best_distance_mean'] * 100
            ).round(2)

            stability = stability.reset_index()
            stability.insert(0, 'run', run_number)

            all_stability.append(stability)

        except Exception as e:
            print(f"✗ Error processing {file.name}: {e}")

    if not all_stability:
        print("❌ Failed to calculate stability statistics!")
        return best_runs_df, None, None

    stability_df = pd.concat(all_stability, ignore_index=True)

    # Merge datasets: stability + parameters
    param_columns = ['run', 'experiment_name', 'description', 'population_size',
                     'num_generations', 'mutation_percent_genes', 'crossover_type',
                     'parent_selection_type', 'K_tournament']

    param_columns = [col for col in param_columns if col in best_runs_df.columns]
    params_df = best_runs_df[param_columns].copy()

    full_df = stability_df.merge(params_df, on=['run', 'experiment_name'], how='left')

    print(f"\n✅ Collected {len(full_df)} rows of data")

    return best_runs_df, stability_df, full_df


def analyze_rankings(full_df):
    """
    Generate and display experiment rankings from multiple perspectives.

    Creates three complementary rankings:
    1. Best Quality: Lowest mean distance (best solutions)
    2. Best Stability: Lowest CV (most reproducible)
    3. Best Compromise: Balance between quality and stability

    The compromise score normalizes both metrics to [0,1] and averages them,
    favoring configurations that perform well on both dimensions.

    Args:
        full_df: DataFrame with aggregated experiment data
    """

    print("\n" + "=" * 80)
    print("STEP 2: EXPERIMENT RANKINGS")
    print("=" * 80 + "\n")

    # Ranking 1: Best Average Results
    print("🏆 TOP 10 - BEST AVERAGE RESULTS")
    print("-" * 80)

    top_mean = full_df.nsmallest(10, 'best_distance_mean')
    for idx, row in enumerate(top_mean.itertuples(), 1):
        desc = f" ({row.description})" if hasattr(row, 'description') and pd.notna(row.description) else ""
        print(f"{idx:2d}. {row.run:8s} | {row.experiment_name:20s}{desc}")
        print(f"     Mean: {row.best_distance_mean:6.2f} | CV: {row.cv_distance:5.2f}%")

    # Ranking 2: Most Stable
    print("\n" + "-" * 80)
    print("✅ TOP 10 - MOST STABLE (lowest CV)")
    print("-" * 80)

    top_stable = full_df.nsmallest(10, 'cv_distance')
    for idx, row in enumerate(top_stable.itertuples(), 1):
        desc = f" ({row.description})" if hasattr(row, 'description') and pd.notna(row.description) else ""
        print(f"{idx:2d}. {row.run:8s} | {row.experiment_name:20s}{desc}")
        print(f"     CV: {row.cv_distance:5.2f}% | Mean: {row.best_distance_mean:6.2f}")

    # Ranking 3: Best Compromise
    print("\n" + "-" * 80)
    print("⚖️  TOP 10 - BEST COMPROMISE (quality + stability)")
    print("-" * 80)

    # Normalize metrics to [0,1] scale for fair comparison
    full_df['norm_mean'] = (
            (full_df['best_distance_mean'] - full_df['best_distance_mean'].min()) /
            (full_df['best_distance_mean'].max() - full_df['best_distance_mean'].min())
    )
    full_df['norm_cv'] = (
            (full_df['cv_distance'] - full_df['cv_distance'].min()) /
            (full_df['cv_distance'].max() - full_df['cv_distance'].min())
    )
    # Lower compromise score = better overall performance
    full_df['compromise_score'] = (full_df['norm_mean'] + full_df['norm_cv']) / 2

    top_compromise = full_df.nsmallest(10, 'compromise_score')
    for idx, row in enumerate(top_compromise.itertuples(), 1):
        desc = f" ({row.description})" if hasattr(row, 'description') and pd.notna(row.description) else ""
        print(f"{idx:2d}. {row.run:8s} | {row.experiment_name:20s}{desc}")
        print(f"     Mean: {row.best_distance_mean:6.2f} | CV: {row.cv_distance:5.2f}%")


# ============================================================================
# CHART GENERATION FUNCTIONS
# ============================================================================

def chart1_top10_best_results(best_runs_df, charts_dir):
    """
    Chart 1: TOP 10 best individual results with dual axes.

    Shows both route distance (bars) and generation count (line) to help
    identify configurations that find good solutions efficiently.

    Args:
        best_runs_df: DataFrame with best result per (run × experiment)
        charts_dir: Output directory for chart
    """

    print("\n📊 Generating: chart1_top10_best_results.png")

    top10 = best_runs_df.nsmallest(10, 'best_distance').copy()
    top10['label'] = top10['run'] + ': ' + top10['experiment_name']

    fig, ax1 = plt.subplots(figsize=(16, 10))

    colors_bars = [COLORS['extended_palette'][i % len(COLORS['extended_palette'])]
                   for i in range(len(top10))]

    # Left axis: Distance (bars)
    ax1.barh(range(len(top10)), top10['best_distance'],
             color=colors_bars, alpha=0.8, edgecolor='white', linewidth=2)

    # Add value labels
    for i, (idx, row) in enumerate(top10.iterrows()):
        ax1.text(row['best_distance'] + 5, i, f"{row['best_distance']:.2f}",
                 va='center', fontweight='bold', fontsize=13)

    ax1.set_yticks(range(len(top10)))
    ax1.set_yticklabels(top10['label'], fontsize=12)
    ax1.set_xlabel('Best route distance', fontsize=16, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()

    # Reference line: Optimal solution
    optimal = 386.33  # Known optimal for this TSP instance
    ax1.axvline(optimal, color='green', linestyle='--', linewidth=3,
                alpha=0.7, label=f'Optimum: {optimal}')

    # Right axis: Generation count (line)
    ax2 = ax1.twiny()
    ax2.plot(top10['generations_completed'].values, range(len(top10)),
             color=COLORS['accent'], marker='o', markersize=10, linewidth=3,
             alpha=0.9, zorder=10)

    ax2.set_xlabel('Number of generations', fontsize=16, fontweight='bold', color=COLORS['accent'])
    ax2.tick_params(axis='x', labelcolor=COLORS['accent'])

    ax1.set_title('🏆 TOP 10 Best Individual Results',
                  fontsize=22, fontweight='bold', pad=60)
    ax1.legend(fontsize=13)

    plt.tight_layout()
    plt.savefig(charts_dir / 'chart1_top10_best_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✓ Saved")


def chart2_stability_vs_quality(full_df, charts_dir):
    """
    Chart 2: Scatter plot showing stability vs quality tradeoff.

    Each point represents one (run × experiment) combination.
    Ideal configurations appear in the lower-left corner (low distance, low CV).
    Reference lines show median values for context.

    Args:
        full_df: DataFrame with aggregated experiment data
        charts_dir: Output directory for chart
    """

    print("\n📊 Generating: chart2_stability_vs_quality.png")

    fig, ax = plt.subplots(figsize=(16, 12))

    # Color-code by experiment name for easy identification
    unique_experiments = full_df['experiment_name'].unique()
    color_map = {exp: COLORS['extended_palette'][i % len(COLORS['extended_palette'])]
                 for i, exp in enumerate(unique_experiments)}

    colors = [color_map[exp] for exp in full_df['experiment_name']]

    ax.scatter(full_df['best_distance_mean'], full_df['cv_distance'],
               s=200, alpha=0.7, c=colors, edgecolors='white', linewidth=2)

    # Label each point
    for idx, row in full_df.iterrows():
        label = f"{row['run']}\n{row['experiment_name']}"
        ax.annotate(label, (row['best_distance_mean'], row['cv_distance']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Add reference lines at median values
    median_mean = full_df['best_distance_mean'].median()
    median_cv = full_df['cv_distance'].median()

    ax.axhline(median_cv, color=COLORS['secondary'], linestyle='--',
               alpha=0.5, linewidth=2, label=f'Median CV: {median_cv:.1f}%')
    ax.axvline(median_mean, color=COLORS['primary'], linestyle='--',
               alpha=0.5, linewidth=2, label=f'Median mean: {median_mean:.1f}')

    ax.set_xlabel('Average route distance (lower = better)', fontsize=18, fontweight='bold')
    ax.set_ylabel('CV% (lower = more stable)', fontsize=18, fontweight='bold')
    ax.set_title('📊 Stability vs Quality Tradeoff', fontsize=22, fontweight='bold', pad=20)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(charts_dir / 'chart2_stability_vs_quality.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✓ Saved")


def chart3_three_perspectives(full_df, charts_dir):
    """
    Chart 3: Three complementary rankings (TOP 5 each).

    Side-by-side comparison of:
    - Best quality (lowest mean distance)
    - Best stability (lowest CV)
    - Best compromise (balanced score)

    Helps identify whether a single configuration excels at all metrics
    or if different configurations are optimal for different objectives.

    Args:
        full_df: DataFrame with aggregated experiment data
        charts_dir: Output directory for chart
    """

    print("\n📊 Generating: chart3_three_perspectives.png")

    top_quality = full_df.nsmallest(5, 'best_distance_mean')
    top_stable = full_df.nsmallest(5, 'cv_distance')
    top_compromise = full_df.nsmallest(5, 'compromise_score')

    fig, axes = plt.subplots(1, 3, figsize=(24, 10))

    def plot_ranking(ax, data, metric, title, xlabel, color):
        """Helper function to create consistent ranking bars."""
        data = data.copy()
        data['label'] = data['run'] + '\n' + data['experiment_name']

        ax.barh(range(len(data)), data[metric],
                color=color, alpha=0.8, edgecolor='white', linewidth=2)

        for i, (idx, row) in enumerate(data.iterrows()):
            ax.text(row[metric] + row[metric] * 0.02, i, f"{row[metric]:.2f}",
                    va='center', fontweight='bold', fontsize=12)

        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data['label'], fontsize=11)
        ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

    plot_ranking(axes[0], top_quality, 'best_distance_mean',
                 '🏆 TOP 5 Best Quality', 'Average distance', COLORS['primary'])
    plot_ranking(axes[1], top_stable, 'cv_distance',
                 '✅ TOP 5 Most Stable', 'CV (%)', COLORS['accent'])
    plot_ranking(axes[2], top_compromise, 'compromise_score',
                 '⚖️ TOP 5 Best Compromise', 'Score', COLORS['secondary'])

    plt.suptitle('Three Perspectives of Experiment Analysis', fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(charts_dir / 'chart3_three_perspectives.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✓ Saved")


def chart4_cv_distribution(full_df, charts_dir):
    """
    Chart 4: Distribution of Coefficient of Variation (CV).

    Histogram showing how many experiments fall into each stability category:
    - Very stable: CV < 5%
    - Moderately stable: CV 5-10%
    - Unstable: CV > 10%

    Reference lines mark these thresholds and the median/mean values.

    Args:
        full_df: DataFrame with aggregated experiment data
        charts_dir: Output directory for chart
    """

    print("\n📊 Generating: chart4_cv_distribution.png")

    fig, ax = plt.subplots(figsize=(14, 10))

    ax.hist(full_df['cv_distance'], bins=20, alpha=0.6, color=COLORS['primary'],
            edgecolor='white', linewidth=2)

    median_cv = full_df['cv_distance'].median()
    mean_cv = full_df['cv_distance'].mean()

    # Threshold reference lines
    ax.axvline(5, color='green', linestyle='--', linewidth=3, label='Very stable (CV=5%)')
    ax.axvline(10, color='orange', linestyle='--', linewidth=3, label='Moderate (CV=10%)')
    ax.axvline(median_cv, color='red', linestyle='-', linewidth=3,
               label=f'Median: {median_cv:.1f}%')

    ax.set_xlabel('Coefficient of Variation CV (%)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Number of experiments', fontsize=18, fontweight='bold')
    ax.set_title('📊 Distribution of Experiment Stability', fontsize=22, fontweight='bold', pad=20)
    ax.legend(fontsize=14)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(charts_dir / 'chart4_cv_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✓ Saved")


def chart5_parameter_comparison(full_df, charts_dir):
    """
    Chart 5: Parameter impact analysis (2×2 grid).

    Compares how different GA parameters affect:
    - Quality (mean distance)
    - Stability (CV)

    Currently analyzes:
    - Crossover type (e.g., single-point, two-point, OX)
    - Parent selection type (e.g., tournament, roulette, rank)

    Args:
        full_df: DataFrame with aggregated experiment data
        charts_dir: Output directory for chart
    """

    print("\n📊 Generating: chart5_parameter_comparison.png")

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # Analyze crossover type impact
    if 'crossover_type' in full_df.columns and full_df['crossover_type'].notna().any():
        stats = full_df.groupby('crossover_type').agg({
            'best_distance_mean': 'mean',
            'cv_distance': 'mean'
        }).round(2)

        # Subplot 1: Crossover - Quality
        ax = axes[0, 0]
        stats['best_distance_mean'].plot(kind='bar', ax=ax, color=COLORS['primary'], alpha=0.8)
        ax.set_title('Crossover Type - Quality Impact', fontsize=16, fontweight='bold')
        ax.set_ylabel('Average distance', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Subplot 3: Crossover - Stability
        ax = axes[1, 0]
        stats['cv_distance'].plot(kind='bar', ax=ax, color=COLORS['secondary'], alpha=0.8)
        ax.set_title('Crossover Type - Stability Impact', fontsize=16, fontweight='bold')
        ax.set_ylabel('CV (%)', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(5, color='green', linestyle='--', alpha=0.5, label='Very stable')
        ax.axhline(10, color='orange', linestyle='--', alpha=0.5, label='Moderate')
        ax.legend()

    # Analyze selection type impact
    if 'parent_selection_type' in full_df.columns and full_df['parent_selection_type'].notna().any():
        stats2 = full_df.groupby('parent_selection_type').agg({
            'best_distance_mean': 'mean',
            'cv_distance': 'mean'
        }).round(2)

        # Subplot 2: Selection - Quality
        ax = axes[0, 1]
        stats2['best_distance_mean'].plot(kind='bar', ax=ax, color=COLORS['accent'], alpha=0.8)
        ax.set_title('Selection Type - Quality Impact', fontsize=16, fontweight='bold')
        ax.set_ylabel('Average distance', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Subplot 4: Selection - Stability
        ax = axes[1, 1]
        stats2['cv_distance'].plot(kind='bar', ax=ax, color=COLORS['accent'], alpha=0.8)
        ax.set_title('Selection Type - Stability Impact', fontsize=16, fontweight='bold')
        ax.set_ylabel('CV (%)', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(5, color='green', linestyle='--', alpha=0.5)
        ax.axhline(10, color='orange', linestyle='--', alpha=0.5)

    plt.suptitle('Parameter Impact on Performance', fontsize=24, fontweight='bold')
    plt.tight_layout()
    plt.savefig(charts_dir / 'chart5_parameter_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✓ Saved")


def chart6_boxplot_all(experiments_dir, charts_dir):
    """
    Chart 6: Boxplot comparing all experiments (raw data).

    Shows the distribution of ALL trial results (not just best per run).
    Boxplots reveal:
    - Median performance (middle line)
    - Spread/variance (box width)
    - Outliers (individual points)

    Experiments are sorted by median distance for easier comparison.

    Args:
        experiments_dir: Directory containing raw experiment CSV files
        charts_dir: Output directory for chart
    """

    print("\n📊 Generating: chart6_boxplot_all.png")

    base_path = Path(experiments_dir)
    result_files = sorted(base_path.glob('experiment_results_run_*.csv'))

    # Load and combine all raw data
    all_data = []
    for file in result_files:
        df = pd.read_csv(file)
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)

    fig, ax = plt.subplots(figsize=(18, 12))

    # Sort experiments by median for better readability
    exp_order = combined.groupby('experiment_name')['best_distance'].median().sort_values().index
    data_for_box = [combined[combined['experiment_name'] == exp]['best_distance'].values
                    for exp in exp_order]

    colors = [COLORS['extended_palette'][i % len(COLORS['extended_palette'])]
              for i in range(len(exp_order))]

    # Create boxplots
    bp = ax.boxplot(data_for_box, labels=exp_order, vert=False, patch_artist=True,
                    boxprops=dict(linewidth=2.5),
                    whiskerprops=dict(linewidth=2.5),
                    medianprops=dict(linewidth=4, color='white'))

    # Apply colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Route distance', fontsize=18, fontweight='bold')
    ax.set_ylabel('Experiment', fontsize=18, fontweight='bold')
    ax.set_title('📦 Comparison of All Experiments (All Trials)', fontsize=22, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(charts_dir / 'chart6_boxplot_all.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✓ Saved")


def chart7_heatmap(full_df, charts_dir):
    """
    Chart 7: Consistency heatmap across runs.

    Matrix view where:
    - Rows: Different experiments
    - Columns: Different runs
    - Color: Mean distance (darker = better)

    Reveals which experiments perform consistently well across multiple runs
    and which are more variable.

    Args:
        full_df: DataFrame with aggregated experiment data
        charts_dir: Output directory for chart
    """

    print("\n📊 Generating: chart7_heatmap.png")

    # Create pivot table: experiments × runs
    pivot = full_df.pivot(index='experiment_name', columns='run', values='best_distance_mean')

    # Sort columns numerically and rows by average performance
    pivot = pivot[sorted(pivot.columns, key=lambda x: int(x.replace('run_', '')))]
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(20, 12))

    # Create heatmap (reversed colormap: darker = better)
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average distance', fontsize=16, fontweight='bold')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=12)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=12)

    ax.set_xlabel('Run', fontsize=18, fontweight='bold')
    ax.set_ylabel('Experiment', fontsize=18, fontweight='bold')
    ax.set_title('🔥 Experiment Consistency Heatmap', fontsize=22, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(charts_dir / 'chart7_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✓ Saved")


def generate_summary_report(best_runs_df, full_df, output_dir):
    """
    Generate comprehensive text report with analysis and recommendations.

    Report includes:
    - Basic statistics (runs, experiments, data points)
    - Best overall result with full parameters
    - TOP 3 rankings (quality, stability, compromise)
    - Stability analysis with actionable insights
    - Parameter comparison (if available)
    - Educational conclusions for students

    Args:
        best_runs_df: DataFrame with best result per (run × experiment)
        full_df: DataFrame with aggregated experiment data
        output_dir: Output directory for report file
    """

    print("\n" + "=" * 80)
    print("STEP 4: GENERATING SUMMARY REPORT")
    print("=" * 80)

    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE SUMMARY REPORT")
    report.append("=" * 80)
    report.append("")

    # Basic statistics
    report.append("BASIC STATISTICS:")
    report.append(f"  Number of runs: {full_df['run'].nunique()}")
    report.append(f"  Number of experiments: {full_df['experiment_name'].nunique()}")
    report.append(f"  Total data points: {len(full_df)}")
    report.append("")

    # Overall best result
    best = best_runs_df.loc[best_runs_df['best_distance'].idxmin()]
    report.append("🥇 OVERALL BEST RESULT:")
    report.append(f"  Run: {best['run']}")
    report.append(f"  Experiment: {best['experiment_name']}")
    report.append(f"  Distance: {best['best_distance']:.2f}")
    report.append("")

    # TOP 3 rankings
    report.append("🏆 TOP 3 - BEST AVERAGE RESULTS:")
    for idx, row in enumerate(full_df.nsmallest(3, 'best_distance_mean').itertuples(), 1):
        report.append(f"  {idx}. {row.run} - {row.experiment_name}: {row.best_distance_mean:.2f}")
    report.append("")

    # Stability summary
    report.append("📊 STABILITY ANALYSIS:")
    avg_cv = full_df['cv_distance'].mean()
    report.append(f"  Average CV: {avg_cv:.2f}%")
    very_stable = (full_df['cv_distance'] < 5).sum()
    report.append(f"  Very stable (CV<5%): {very_stable} ({very_stable / len(full_df) * 100:.1f}%)")
    report.append("")

    # Educational conclusions
    report.append("=" * 80)
    report.append("💡 KEY TAKEAWAYS FOR STUDENTS:")
    report.append("=" * 80)
    report.append("")
    report.append("1. GENETIC ALGORITHMS ARE STOCHASTIC")
    report.append("   - Same parameters yield different results across runs")
    report.append("   - ALWAYS report statistics (mean, std, CV) not single results")
    report.append("")
    report.append("2. STABILITY MATTERS")
    report.append("   - CV < 5%: Excellent reproducibility")
    report.append("   - CV > 10%: Needs more trials or better parameters")
    report.append("")
    report.append("3. COEFFICIENT OF VARIATION (CV)")
    report.append("   - CV = (std / mean) × 100%")
    report.append("   - Enables fair comparison between different experiments")
    report.append("")

    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    # Save report
    report_text = "\n".join(report)

    report_file = Path(output_dir) / 'summary_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n✅ Saved: summary_report.txt")
    print("\n" + report_text)