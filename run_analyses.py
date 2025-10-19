"""
run_analyses.py - Multi-run experiment analysis pipeline

This script orchestrates comprehensive analysis of multiple GA experiment runs.
It collects data from multiple CSV files, calculates aggregate statistics,
generates comparative visualizations, and produces a summary report.

Use this when you have multiple experiment runs stored as:
  experiments/experiment_results_run_1.csv
  experiments/experiment_results_run_2.csv
  etc.
"""

from pathlib import Path
from utils_unified_analyses import (
    collect_all_experiment_data,
    analyze_rankings,
    chart1_top10_best_results,
    chart2_stability_vs_quality,
    chart3_three_perspectives,
    chart4_cv_distribution,
    chart5_parameter_comparison,
    chart6_boxplot_all,
    chart7_heatmap,
    generate_summary_report
)

# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERIMENTS_DIR = 'experiments'  # Folder containing experiment_results_run_*.csv files
OUTPUT_DIR = 'analysis_results'  # Output folder for all results
DATA_DIR = Path(OUTPUT_DIR) / 'data'      # Aggregated CSV files
CHARTS_DIR = Path(OUTPUT_DIR) / 'charts'  # PNG visualizations


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Execute complete multi-run analysis pipeline.

    Pipeline steps:
    1. Collect data from all experiment_results_run_*.csv files
    2. Calculate rankings across multiple perspectives
    3. Generate 7 publication-ready charts
    4. Create comprehensive text report
    """

    print("\n" + "=" * 80)
    print("MULTI-RUN EXPERIMENT ANALYSIS - GENETIC ALGORITHMS TSP")
    print("=" * 80)
    print(f"\nData directory: {EXPERIMENTS_DIR}/")
    print(f"Output directory: {OUTPUT_DIR}/")
    print("=" * 80 + "\n")

    # Create output directory structure
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created folders:")
    print(f"  - {DATA_DIR}/")
    print(f"  - {CHARTS_DIR}/\n")

    # ========================================================================
    # STEP 1: Data Collection
    # ========================================================================

    best_runs_df, stability_df, full_df = collect_all_experiment_data(EXPERIMENTS_DIR)

    if best_runs_df is None:
        print("\n❌ Analysis aborted - no data found")
        return

    # Save aggregated data to CSV
    best_runs_df.to_csv(DATA_DIR / 'best_runs_per_experiment.csv', index=False)
    stability_df.to_csv(DATA_DIR / 'stability_per_run.csv', index=False)
    full_df.to_csv(DATA_DIR / 'full_experiment_data.csv', index=False)

    print(f"\n✅ Saved CSV files to {DATA_DIR}/")

    # ========================================================================
    # STEP 2: Rankings Analysis
    # ========================================================================

    analyze_rankings(full_df)

    # ========================================================================
    # STEP 3: Chart Generation
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 3: GENERATING CHARTS")
    print("=" * 80)

    chart1_top10_best_results(best_runs_df, CHARTS_DIR)
    chart2_stability_vs_quality(full_df, CHARTS_DIR)
    chart3_three_perspectives(full_df, CHARTS_DIR)
    chart4_cv_distribution(full_df, CHARTS_DIR)
    chart5_parameter_comparison(full_df, CHARTS_DIR)
    chart6_boxplot_all(EXPERIMENTS_DIR, CHARTS_DIR)
    chart7_heatmap(full_df, CHARTS_DIR)

    print(f"\n✅ All charts saved to {CHARTS_DIR}/")

    # ========================================================================
    # STEP 4: Summary Report
    # ========================================================================

    generate_summary_report(best_runs_df, full_df, OUTPUT_DIR)

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\n📁 Generated:")
    print(f"  • 3 CSV files in {DATA_DIR}/")
    print(f"  • 7 PNG charts in {CHARTS_DIR}/")
    print(f"  • 1 TXT report in {OUTPUT_DIR}/")
    print("\n📄 File list:")
    print("  CSV:")
    print("    - best_runs_per_experiment.csv")
    print("    - stability_per_run.csv")
    print("    - full_experiment_data.csv")
    print("  Charts:")
    print("    - chart1_top10_best_results.png")
    print("    - chart2_stability_vs_quality.png")
    print("    - chart3_three_perspectives.png")
    print("    - chart4_cv_distribution.png")
    print("    - chart5_parameter_comparison.png")
    print("    - chart6_boxplot_all.png")
    print("    - chart7_heatmap.png")
    print("  Report:")
    print("    - summary_report.txt")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()