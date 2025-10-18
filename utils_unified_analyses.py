"""
Funkcje pomocnicze do analizy eksperymentów algorytmów genetycznych TSP
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# ============================================================================
# KONFIGURACJA
# ============================================================================

COLORS = {
    'primary': '#5c26ff',
    'secondary': '#9c31ff',
    'accent': '#ffba0a',
    'extended_palette': [
        '#5c26ff', '#ffba0a', '#ff6b6b', '#4ecdc4', '#95e1d3',
        '#c7a8ff', '#ffd93d', '#ff8fab', '#6bcf7f', '#a8daff',
        '#ffb347', '#b19cd9', '#77dd77', '#ff6f91', '#aec6cf'
    ]
}

# Konfiguracja matplotlib
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
# FUNKCJE ZBIERANIA DANYCH
# ============================================================================

def collect_all_experiment_data(experiments_dir='experiments'):
    """
    Zbiera wszystkie dane z eksperymentów:
    1. Najlepsze wyniki dla każdego (run, experiment)
    2. Statystyki stabilności dla każdego (run, experiment)
    3. Łączy obie tabele w jedną

    Returns:
        tuple: (best_runs_df, stability_df, full_df) lub (None, None, None) w przypadku błędu
    """

    base_path = Path(experiments_dir)

    print("=" * 80)
    print("KROK 1: ZBIERANIE DANYCH Z EKSPERYMENTÓW")
    print("=" * 80)

    result_files = sorted(base_path.glob('experiment_results_run_*.csv'))

    if not result_files:
        print(f"❌ Nie znaleziono plików experiment_results_run_*.csv w {experiments_dir}/")
        return None, None, None

    print(f"\nZnaleziono {len(result_files)} plików z wynikami\n")

    # Zbierz najlepsze wyniki
    all_best_runs = []

    for file in result_files:
        run_number = file.stem.replace('experiment_results_', '')

        try:
            df = pd.read_csv(file)
            best_per_experiment = df.loc[df.groupby('experiment_name')['best_distance'].idxmin()].copy()

            columns_to_drop = ['timestamp', 'run_id']
            best_per_experiment = best_per_experiment.drop(
                columns=[col for col in columns_to_drop if col in best_per_experiment.columns]
            )

            best_per_experiment.insert(0, 'run', run_number)
            all_best_runs.append(best_per_experiment)

            print(f"✓ {run_number}: {len(best_per_experiment)} eksperymentów")

        except Exception as e:
            print(f"✗ Błąd przy {file.name}: {e}")

    if not all_best_runs:
        print("❌ Nie udało się zebrać danych!")
        return None, None, None

    best_runs_df = pd.concat(all_best_runs, ignore_index=True)

    # Oblicz statystyki stabilności
    all_stability = []

    for file in result_files:
        run_number = file.stem.replace('experiment_results_', '')

        try:
            df = pd.read_csv(file)

            stability = df.groupby('experiment_name').agg({
                'best_distance': ['count', 'mean', 'std', 'min', 'max'],
                'generations_completed': ['mean', 'std']
            }).round(2)

            stability.columns = ['_'.join(col).strip() for col in stability.columns]
            stability['cv_distance'] = (
                    stability['best_distance_std'] / stability['best_distance_mean'] * 100
            ).round(2)

            stability = stability.reset_index()
            stability.insert(0, 'run', run_number)

            all_stability.append(stability)

        except Exception as e:
            print(f"✗ Błąd przy {file.name}: {e}")

    if not all_stability:
        print("❌ Nie udało się obliczyć statystyk!")
        return best_runs_df, None, None

    stability_df = pd.concat(all_stability, ignore_index=True)

    # Połącz dane
    param_columns = ['run', 'experiment_name', 'description', 'population_size',
                     'num_generations', 'mutation_percent_genes', 'crossover_type',
                     'parent_selection_type', 'K_tournament']

    param_columns = [col for col in param_columns if col in best_runs_df.columns]
    params_df = best_runs_df[param_columns].copy()

    full_df = stability_df.merge(params_df, on=['run', 'experiment_name'], how='left')

    print(f"\n✅ Zebrano {len(full_df)} wierszy danych")

    return best_runs_df, stability_df, full_df


def analyze_rankings(full_df):
    """Wyświetla rankingi eksperymentów"""

    print("\n" + "=" * 80)
    print("KROK 2: RANKINGI EKSPERYMENTÓW")
    print("=" * 80 + "\n")

    # TOP 10 najlepsze średnie
    print("🏆 TOP 10 - NAJLEPSZE ŚREDNIE WYNIKI")
    print("-" * 80)

    top_mean = full_df.nsmallest(10, 'best_distance_mean')
    for idx, row in enumerate(top_mean.itertuples(), 1):
        desc = f" ({row.description})" if hasattr(row, 'description') and pd.notna(row.description) else ""
        print(f"{idx:2d}. {row.run:8s} | {row.experiment_name:20s}{desc}")
        print(f"     Średnia: {row.best_distance_mean:6.2f} | CV: {row.cv_distance:5.2f}%")

    # TOP 10 najbardziej stabilne
    print("\n" + "-" * 80)
    print("✅ TOP 10 - NAJBARDZIEJ STABILNE (najmniejszy CV)")
    print("-" * 80)

    top_stable = full_df.nsmallest(10, 'cv_distance')
    for idx, row in enumerate(top_stable.itertuples(), 1):
        desc = f" ({row.description})" if hasattr(row, 'description') and pd.notna(row.description) else ""
        print(f"{idx:2d}. {row.run:8s} | {row.experiment_name:20s}{desc}")
        print(f"     CV: {row.cv_distance:5.2f}% | Średnia: {row.best_distance_mean:6.2f}")

    # TOP 10 kompromis
    print("\n" + "-" * 80)
    print("⚖️  TOP 10 - NAJLEPSZY KOMPROMIS")
    print("-" * 80)

    full_df['norm_mean'] = (
            (full_df['best_distance_mean'] - full_df['best_distance_mean'].min()) /
            (full_df['best_distance_mean'].max() - full_df['best_distance_mean'].min())
    )
    full_df['norm_cv'] = (
            (full_df['cv_distance'] - full_df['cv_distance'].min()) /
            (full_df['cv_distance'].max() - full_df['cv_distance'].min())
    )
    full_df['compromise_score'] = (full_df['norm_mean'] + full_df['norm_cv']) / 2

    top_compromise = full_df.nsmallest(10, 'compromise_score')
    for idx, row in enumerate(top_compromise.itertuples(), 1):
        desc = f" ({row.description})" if hasattr(row, 'description') and pd.notna(row.description) else ""
        print(f"{idx:2d}. {row.run:8s} | {row.experiment_name:20s}{desc}")
        print(f"     Średnia: {row.best_distance_mean:6.2f} | CV: {row.cv_distance:5.2f}%")


# ============================================================================
# FUNKCJE GENEROWANIA WYKRESÓW
# ============================================================================

def chart1_top10_best_results(best_runs_df, charts_dir):
    """TOP 10 najlepszych pojedynczych wyników"""

    print("\n📊 Generowanie: chart1_top10_best_results.png")

    top10 = best_runs_df.nsmallest(10, 'best_distance').copy()
    top10['label'] = top10['run'] + ': ' + top10['experiment_name']

    fig, ax1 = plt.subplots(figsize=(16, 10))

    colors_bars = [COLORS['extended_palette'][i % len(COLORS['extended_palette'])]
                   for i in range(len(top10))]

    ax1.barh(range(len(top10)), top10['best_distance'],
             color=colors_bars, alpha=0.8, edgecolor='white', linewidth=2)

    for i, (idx, row) in enumerate(top10.iterrows()):
        ax1.text(row['best_distance'] + 5, i, f"{row['best_distance']:.2f}",
                 va='center', fontweight='bold', fontsize=13)

    ax1.set_yticks(range(len(top10)))
    ax1.set_yticklabels(top10['label'], fontsize=12)
    ax1.set_xlabel('Długość najlepszej trasy', fontsize=16, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()

    optimal = 386.33
    ax1.axvline(optimal, color='green', linestyle='--', linewidth=3,
                alpha=0.7, label=f'Optimum: {optimal}')

    ax2 = ax1.twiny()
    ax2.plot(top10['generations_completed'].values, range(len(top10)),
             color=COLORS['accent'], marker='o', markersize=10, linewidth=3,
             alpha=0.9, zorder=10)

    ax2.set_xlabel('Liczba generacji', fontsize=16, fontweight='bold', color=COLORS['accent'])
    ax2.tick_params(axis='x', labelcolor=COLORS['accent'])

    ax1.set_title('🏆 TOP 10 Najlepszych Pojedynczych Wyników',
                  fontsize=22, fontweight='bold', pad=60)
    ax1.legend(fontsize=13)

    plt.tight_layout()
    plt.savefig(charts_dir / 'chart1_top10_best_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✓ Zapisano")


def chart2_stability_vs_quality(full_df, charts_dir):
    """Scatter plot - Stabilność vs Jakość"""

    print("\n📊 Generowanie: chart2_stability_vs_quality.png")

    fig, ax = plt.subplots(figsize=(16, 12))

    unique_experiments = full_df['experiment_name'].unique()
    color_map = {exp: COLORS['extended_palette'][i % len(COLORS['extended_palette'])]
                 for i, exp in enumerate(unique_experiments)}

    colors = [color_map[exp] for exp in full_df['experiment_name']]

    ax.scatter(full_df['best_distance_mean'], full_df['cv_distance'],
               s=200, alpha=0.7, c=colors, edgecolors='white', linewidth=2)

    for idx, row in full_df.iterrows():
        label = f"{row['run']}\n{row['experiment_name']}"
        ax.annotate(label, (row['best_distance_mean'], row['cv_distance']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    median_mean = full_df['best_distance_mean'].median()
    median_cv = full_df['cv_distance'].median()

    ax.axhline(median_cv, color=COLORS['secondary'], linestyle='--',
               alpha=0.5, linewidth=2, label=f'Mediana CV: {median_cv:.1f}%')
    ax.axvline(median_mean, color=COLORS['primary'], linestyle='--',
               alpha=0.5, linewidth=2, label=f'Mediana mean: {median_mean:.1f}')

    ax.set_xlabel('Średnia długość trasy (niższa = lepiej)', fontsize=18, fontweight='bold')
    ax.set_ylabel('CV% (niższy = stabilniej)', fontsize=18, fontweight='bold')
    ax.set_title('📊 Stabilność vs Jakość', fontsize=22, fontweight='bold', pad=20)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(charts_dir / 'chart2_stability_vs_quality.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✓ Zapisano")


def chart3_three_perspectives(full_df, charts_dir):
    """Trzy perspektywy TOP 5"""

    print("\n📊 Generowanie: chart3_three_perspectives.png")

    top_quality = full_df.nsmallest(5, 'best_distance_mean')
    top_stable = full_df.nsmallest(5, 'cv_distance')
    top_compromise = full_df.nsmallest(5, 'compromise_score')

    fig, axes = plt.subplots(1, 3, figsize=(24, 10))

    def plot_ranking(ax, data, metric, title, xlabel, color):
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
                 '🏆 TOP 5 Najlepsze', 'Średnia długość', COLORS['primary'])
    plot_ranking(axes[1], top_stable, 'cv_distance',
                 '✅ TOP 5 Stabilne', 'CV (%)', COLORS['accent'])
    plot_ranking(axes[2], top_compromise, 'compromise_score',
                 '⚖️ TOP 5 Kompromis', 'Score', COLORS['secondary'])

    plt.suptitle('Trzy Perspektywy Analizy', fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(charts_dir / 'chart3_three_perspectives.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✓ Zapisano")


def chart4_cv_distribution(full_df, charts_dir):
    """Rozkład CV"""

    print("\n📊 Generowanie: chart4_cv_distribution.png")

    fig, ax = plt.subplots(figsize=(14, 10))

    ax.hist(full_df['cv_distance'], bins=20, alpha=0.6, color=COLORS['primary'],
            edgecolor='white', linewidth=2)

    median_cv = full_df['cv_distance'].median()
    mean_cv = full_df['cv_distance'].mean()

    ax.axvline(5, color='green', linestyle='--', linewidth=3, label='CV=5%')
    ax.axvline(10, color='orange', linestyle='--', linewidth=3, label='CV=10%')
    ax.axvline(median_cv, color='red', linestyle='-', linewidth=3,
               label=f'Mediana: {median_cv:.1f}%')

    ax.set_xlabel('CV (%)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Liczba eksperymentów', fontsize=18, fontweight='bold')
    ax.set_title('📊 Rozkład Stabilności', fontsize=22, fontweight='bold', pad=20)
    ax.legend(fontsize=14)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(charts_dir / 'chart4_cv_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✓ Zapisano")


def chart5_parameter_comparison(full_df, charts_dir):
    """Porównanie parametrów"""

    print("\n📊 Generowanie: chart5_parameter_comparison.png")

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # Crossover - jakość
    if 'crossover_type' in full_df.columns and full_df['crossover_type'].notna().any():
        stats = full_df.groupby('crossover_type').agg({
            'best_distance_mean': 'mean',
            'cv_distance': 'mean'
        }).round(2)

        ax = axes[0, 0]
        stats['best_distance_mean'].plot(kind='bar', ax=ax, color=COLORS['primary'], alpha=0.8)
        ax.set_title('Crossover - Jakość', fontsize=16, fontweight='bold')
        ax.set_ylabel('Średnia długość', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Selection - jakość
        ax = axes[0, 1]
        if 'parent_selection_type' in full_df.columns:
            stats2 = full_df.groupby('parent_selection_type')['best_distance_mean'].mean()
            stats2.plot(kind='bar', ax=ax, color=COLORS['accent'], alpha=0.8)
        ax.set_title('Selekcja - Jakość', fontsize=16, fontweight='bold')
        ax.set_ylabel('Średnia długość', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Crossover - stabilność
        ax = axes[1, 0]
        stats['cv_distance'].plot(kind='bar', ax=ax, color=COLORS['secondary'], alpha=0.8)
        ax.set_title('Crossover - Stabilność', fontsize=16, fontweight='bold')
        ax.set_ylabel('CV (%)', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Selection - stabilność
        ax = axes[1, 1]
        if 'parent_selection_type' in full_df.columns:
            stats3 = full_df.groupby('parent_selection_type')['cv_distance'].mean()
            stats3.plot(kind='bar', ax=ax, color=COLORS['accent'], alpha=0.8)
        ax.set_title('Selekcja - Stabilność', fontsize=16, fontweight='bold')
        ax.set_ylabel('CV (%)', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Wpływ Parametrów', fontsize=24, fontweight='bold')
    plt.tight_layout()
    plt.savefig(charts_dir / 'chart5_parameter_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✓ Zapisano")


def chart6_boxplot_all(experiments_dir, charts_dir):
    """Boxplot wszystkich eksperymentów"""

    print("\n📊 Generowanie: chart6_boxplot_all.png")

    base_path = Path(experiments_dir)
    result_files = sorted(base_path.glob('experiment_results_run_*.csv'))

    all_data = []
    for file in result_files:
        df = pd.read_csv(file)
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)

    fig, ax = plt.subplots(figsize=(18, 12))

    exp_order = combined.groupby('experiment_name')['best_distance'].median().sort_values().index
    data_for_box = [combined[combined['experiment_name'] == exp]['best_distance'].values
                    for exp in exp_order]

    colors = [COLORS['extended_palette'][i % len(COLORS['extended_palette'])]
              for i in range(len(exp_order))]

    bp = ax.boxplot(data_for_box, labels=exp_order, vert=False, patch_artist=True,
                    boxprops=dict(linewidth=2.5),
                    whiskerprops=dict(linewidth=2.5),
                    medianprops=dict(linewidth=4, color='white'))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Długość trasy', fontsize=18, fontweight='bold')
    ax.set_ylabel('Eksperyment', fontsize=18, fontweight='bold')
    ax.set_title('📦 Porównanie Wszystkich Eksperymentów', fontsize=22, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(charts_dir / 'chart6_boxplot_all.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✓ Zapisano")


def chart7_heatmap(full_df, charts_dir):
    """Heatmapa konsystencji"""

    print("\n📊 Generowanie: chart7_heatmap.png")

    pivot = full_df.pivot(index='experiment_name', columns='run', values='best_distance_mean')
    pivot = pivot[sorted(pivot.columns, key=lambda x: int(x.replace('run_', '')))]
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(20, 12))

    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Średnia długość', fontsize=16, fontweight='bold')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=12)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=12)

    ax.set_xlabel('Run', fontsize=18, fontweight='bold')
    ax.set_ylabel('Eksperyment', fontsize=18, fontweight='bold')
    ax.set_title('🔥 Heatmapa Konsystencji', fontsize=22, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(charts_dir / 'chart7_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✓ Zapisano")


def generate_summary_report(best_runs_df, full_df, output_dir):
    """Generuje raport tekstowy"""

    print("\n" + "=" * 80)
    print("KROK 4: RAPORT TEKSTOWY")
    print("=" * 80)

    report = []
    report.append("=" * 80)
    report.append("RAPORT PODSUMOWUJĄCY")
    report.append("=" * 80)
    report.append("")

    # Podstawowe statystyki
    report.append("PODSTAWOWE STATYSTYKI:")
    report.append(f"  Liczba runs: {full_df['run'].nunique()}")
    report.append(f"  Liczba eksperymentów: {full_df['experiment_name'].nunique()}")
    report.append(f"  Łączna liczba: {len(full_df)}")
    report.append("")

    # Najlepszy wynik
    best = best_runs_df.loc[best_runs_df['best_distance'].idxmin()]
    report.append("🥇 NAJLEPSZY WYNIK:")
    report.append(f"  Run: {best['run']}")
    report.append(f"  Eksperyment: {best['experiment_name']}")
    report.append(f"  Distance: {best['best_distance']:.2f}")
    report.append("")

    # TOP 3
    report.append("🏆 TOP 3 - NAJLEPSZE ŚREDNIE:")
    for idx, row in enumerate(full_df.nsmallest(3, 'best_distance_mean').itertuples(), 1):
        report.append(f"  {idx}. {row.run} - {row.experiment_name}: {row.best_distance_mean:.2f}")
    report.append("")

    # Stabilność
    report.append("📊 STABILNOŚĆ:")
    avg_cv = full_df['cv_distance'].mean()
    report.append(f"  Średni CV: {avg_cv:.2f}%")
    very_stable = (full_df['cv_distance'] < 5).sum()
    report.append(f"  Bardzo stabilne (CV<5%): {very_stable} ({very_stable / len(full_df) * 100:.1f}%)")
    report.append("")

    report.append("=" * 80)

    report_text = "\n".join(report)

    report_file = Path(output_dir) / 'summary_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n✅ Zapisano: summary_report.txt")
    print("\n" + report_text)