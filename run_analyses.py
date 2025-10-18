"""
Skrypt uruchamiający pełną analizę eksperymentów algorytmów genetycznych TSP
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
# KONFIGURACJA
# ============================================================================

EXPERIMENTS_DIR = 'experiments'
OUTPUT_DIR = 'analysis_results'
DATA_DIR = Path(OUTPUT_DIR) / 'data'
CHARTS_DIR = Path(OUTPUT_DIR) / 'charts'


# ============================================================================
# GŁÓWNY PIPELINE
# ============================================================================

def main():
    """Uruchamia pełną analizę"""

    print("\n" + "=" * 80)
    print("ANALIZA EKSPERYMENTÓW - ALGORYTMY GENETYCZNE TSP")
    print("=" * 80)
    print(f"\nKatalog z danymi: {EXPERIMENTS_DIR}/")
    print(f"Katalog wynikowy: {OUTPUT_DIR}/")
    print("=" * 80 + "\n")

    # Utwórz strukturę folderów
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Utworzono foldery:")
    print(f"  - {DATA_DIR}/")
    print(f"  - {CHARTS_DIR}/\n")

    # ========================================================================
    # KROK 1: Zbieranie danych
    # ========================================================================

    best_runs_df, stability_df, full_df = collect_all_experiment_data(EXPERIMENTS_DIR)

    if best_runs_df is None:
        print("\n❌ Przerwano analizę - brak danych")
        return

    # Zapisz CSV
    best_runs_df.to_csv(DATA_DIR / 'best_runs_per_experiment.csv', index=False)
    stability_df.to_csv(DATA_DIR / 'stability_per_run.csv', index=False)
    full_df.to_csv(DATA_DIR / 'full_experiment_data.csv', index=False)

    print(f"\n✅ Zapisano pliki CSV w {DATA_DIR}/")

    # ========================================================================
    # KROK 2: Rankingi
    # ========================================================================

    analyze_rankings(full_df)

    # ========================================================================
    # KROK 3: Wykresy
    # ========================================================================

    print("\n" + "=" * 80)
    print("KROK 3: GENEROWANIE WYKRESÓW")
    print("=" * 80)

    chart1_top10_best_results(best_runs_df, CHARTS_DIR)
    chart2_stability_vs_quality(full_df, CHARTS_DIR)
    chart3_three_perspectives(full_df, CHARTS_DIR)
    chart4_cv_distribution(full_df, CHARTS_DIR)
    chart5_parameter_comparison(full_df, CHARTS_DIR)
    chart6_boxplot_all(EXPERIMENTS_DIR, CHARTS_DIR)
    chart7_heatmap(full_df, CHARTS_DIR)

    print(f"\n✅ Wszystkie wykresy zapisane w {CHARTS_DIR}/")

    # ========================================================================
    # KROK 4: Raport
    # ========================================================================

    generate_summary_report(best_runs_df, full_df, OUTPUT_DIR)

    # ========================================================================
    # PODSUMOWANIE
    # ========================================================================

    print("\n" + "=" * 80)
    print("✅ ANALIZA ZAKOŃCZONA POMYŚLNIE!")
    print("=" * 80)
    print(f"\n📁 Wygenerowano:")
    print(f"  • 3 pliki CSV w {DATA_DIR}/")
    print(f"  • 7 wykresów PNG w {CHARTS_DIR}/")
    print(f"  • 1 raport TXT w {OUTPUT_DIR}/")
    print("\n📄 Lista plików:")
    print("  CSV:")
    print("    - best_runs_per_experiment.csv")
    print("    - stability_per_run.csv")
    print("    - full_experiment_data.csv")
    print("  Wykresy:")
    print("    - chart1_top10_best_results.png")
    print("    - chart2_stability_vs_quality.png")
    print("    - chart3_three_perspectives.png")
    print("    - chart4_cv_distribution.png")
    print("    - chart5_parameter_comparison.png")
    print("    - chart6_boxplot_all.png")
    print("    - chart7_heatmap.png")
    print("  Raport:")
    print("    - summary_report.txt")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()