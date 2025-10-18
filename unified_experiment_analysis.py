import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# ============================================================================
# KONFIGURACJA
# ============================================================================

# Konfiguracja kolorów
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

# Konfiguracja ścieżek
EXPERIMENTS_DIR = 'experiments'
OUTPUT_DIR = 'analysis_results'
DATA_DIR = Path(OUTPUT_DIR) / 'data'
CHARTS_DIR = Path(OUTPUT_DIR) / 'charts'


# ============================================================================
# KROK 1: ZBIERANIE DANYCH
# ============================================================================

def collect_all_experiment_data():
    """
    Zbiera WSZYSTKIE dane z eksperymentów:
    1. Najlepsze wyniki dla każdego (run, experiment)
    2. Statystyki stabilności dla każdego (run, experiment)
    3. Łączy obie tabele w jedną z pełnymi informacjami
    """

    base_path = Path(EXPERIMENTS_DIR)

    print("=" * 80)
    print("KROK 1: ZBIERANIE WSZYSTKICH DANYCH Z EKSPERYMENTÓW")
    print("=" * 80)

    # Znajdź wszystkie pliki
    result_files = sorted(base_path.glob('experiment_results_run_*.csv'))

    if not result_files:
        print(f"❌ Nie znaleziono plików experiment_results_run_*.csv w {EXPERIMENTS_DIR}/")
        return None, None, None

    print(f"\nZnaleziono {len(result_files)} plików z wynikami\n")

    # ========================================================================
    # CZĘŚĆ 1A: Zbierz najlepsze pojedyncze wyniki
    # ========================================================================

    print("-" * 80)
    print("CZĘŚĆ 1A: ZBIERANIE NAJLEPSZYCH WYNIKÓW")
    print("-" * 80 + "\n")

    all_best_runs = []

    for file in result_files:
        run_number = file.stem.replace('experiment_results_', '')

        try:
            df = pd.read_csv(file)

            # Dla każdego eksperymentu znajdź najlepszy wynik
            best_per_experiment = df.loc[df.groupby('experiment_name')['best_distance'].idxmin()].copy()

            # Usuń timestamp i run_id
            columns_to_drop = ['timestamp', 'run_id']
            best_per_experiment = best_per_experiment.drop(
                columns=[col for col in columns_to_drop if col in best_per_experiment.columns]
            )

            # Dodaj run na początku
            best_per_experiment.insert(0, 'run', run_number)

            all_best_runs.append(best_per_experiment)

            print(f"✓ {run_number}: {len(best_per_experiment)} eksperymentów")
            for _, row in best_per_experiment.iterrows():
                print(f"    - {row['experiment_name']}: best_distance = {row['best_distance']:.2f}")
            print()

        except Exception as e:
            print(f"✗ Błąd przy {file.name}: {e}\n")

    if not all_best_runs:
        print("❌ Nie udało się zebrać najlepszych wyników!")
        return None, None, None

    # Połącz najlepsze wyniki
    best_runs_df = pd.concat(all_best_runs, ignore_index=True)

    print("-" * 80)
    print(f"✅ Zebrano {len(best_runs_df)} najlepszych wyników")
    print("-" * 80 + "\n")

    # ========================================================================
    # CZĘŚĆ 1B: Oblicz statystyki stabilności
    # ========================================================================

    print("-" * 80)
    print("CZĘŚĆ 1B: OBLICZANIE STATYSTYK STABILNOŚCI")
    print("-" * 80 + "\n")

    all_stability = []

    for file in result_files:
        run_number = file.stem.replace('experiment_results_', '')

        try:
            df = pd.read_csv(file)

            # Dla każdego eksperymentu oblicz statystyki
            stability = df.groupby('experiment_name').agg({
                'best_distance': ['count', 'mean', 'std', 'min', 'max'],
                'generations_completed': ['mean', 'std']
            }).round(2)

            # Spłaszcz nazwy kolumn
            stability.columns = ['_'.join(col).strip() for col in stability.columns]

            # Oblicz współczynnik zmienności
            stability['cv_distance'] = (
                    stability['best_distance_std'] / stability['best_distance_mean'] * 100
            ).round(2)

            # Reset index
            stability = stability.reset_index()

            # Dodaj run na początku
            stability.insert(0, 'run', run_number)

            all_stability.append(stability)

            print(f"✓ {run_number}: {len(stability)} eksperymentów")
            for _, row in stability.iterrows():
                print(f"    - {row['experiment_name']}: mean={row['best_distance_mean']:.2f}, "
                      f"std={row['best_distance_std']:.2f}, CV={row['cv_distance']:.2f}%")
            print()

        except Exception as e:
            print(f"✗ Błąd przy {file.name}: {e}\n")

    if not all_stability:
        print("❌ Nie udało się obliczyć statystyk stabilności!")
        return best_runs_df, None, None

    # Połącz statystyki stabilności
    stability_df = pd.concat(all_stability, ignore_index=True)

    print("-" * 80)
    print(f"✅ Obliczono statystyki dla {len(stability_df)} (run × experiment)")
    print("-" * 80 + "\n")

    # ========================================================================
    # CZĘŚĆ 1C: Połącz obie tabele
    # ========================================================================

    print("-" * 80)
    print("CZĘŚĆ 1C: ŁĄCZENIE DANYCH - PEŁNA TABELA")
    print("-" * 80 + "\n")

    # Wybierz parametry z best_runs_df
    param_columns = ['run', 'experiment_name', 'description', 'population_size',
                     'num_generations', 'mutation_percent_genes', 'crossover_type',
                     'parent_selection_type', 'K_tournament']

    param_columns = [col for col in param_columns if col in best_runs_df.columns]
    params_df = best_runs_df[param_columns].copy()

    # Połącz stability z parametrami
    full_df = stability_df.merge(
        params_df,
        on=['run', 'experiment_name'],
        how='left'
    )

    print(f"✅ Utworzono pełną tabelę z {len(full_df)} wierszy")

    # Pokaż podgląd
    print("\n" + "-" * 80)
    print("Podgląd pełnej tabeli:")
    print("-" * 80)

    display_cols = ['run', 'experiment_name', 'best_distance_mean', 'best_distance_std',
                    'cv_distance', 'best_distance_min', 'best_distance_max']
    if 'description' in full_df.columns:
        display_cols.insert(2, 'description')

    existing_cols = [col for col in display_cols if col in full_df.columns]
    print(full_df[existing_cols].head(10).to_string(index=False))

    return best_runs_df, stability_df, full_df


# ============================================================================
# KROK 2: ANALIZA I RANKINGI
# ============================================================================

def analyze_rankings(full_df):
    """
    Tworzy rankingi eksperymentów
    """

    print("\n" + "=" * 80)
    print("KROK 2: RANKINGI EKSPERYMENTÓW")
    print("=" * 80 + "\n")

    # 1. Najlepsze średnie wyniki
    print("🏆 TOP 10 - NAJLEPSZE ŚREDNIE WYNIKI")
    print("-" * 80)

    top_mean = full_df.nsmallest(10, 'best_distance_mean')
    for idx, row in enumerate(top_mean.itertuples(), 1):
        desc = f" ({row.description})" if hasattr(row, 'description') and pd.notna(row.description) else ""
        print(f"{idx:2d}. {row.run:8s} | {row.experiment_name:20s}{desc}")
        print(f"     Średnia: {row.best_distance_mean:6.2f} | Std: {row.best_distance_std:5.2f} | "
              f"CV: {row.cv_distance:5.2f}%")

    # 2. Najbardziej stabilne
    print("\n" + "-" * 80)
    print("✅ TOP 10 - NAJBARDZIEJ STABILNE (najmniejszy CV)")
    print("-" * 80)

    top_stable = full_df.nsmallest(10, 'cv_distance')
    for idx, row in enumerate(top_stable.itertuples(), 1):
        desc = f" ({row.description})" if hasattr(row, 'description') and pd.notna(row.description) else ""
        print(f"{idx:2d}. {row.run:8s} | {row.experiment_name:20s}{desc}")
        print(f"     CV: {row.cv_distance:5.2f}% | Średnia: {row.best_distance_mean:6.2f} | "
              f"Std: {row.best_distance_std:5.2f}")

    # 3. Kompromis
    print("\n" + "-" * 80)
    print("⚖️  TOP 10 - NAJLEPSZY KOMPROMIS (jakość + stabilność)")
    print("-" * 80)

    # Normalizacja
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
        print(f"     Średnia: {row.best_distance_mean:6.2f} | CV: {row.cv_distance:5.2f}% | "
              f"Score: {row.compromise_score:.3f}")


# ============================================================================
# KROK 3: GENEROWANIE WYKRESÓW
# ============================================================================

def chart1_top10_best_results(best_runs_df):
    """
    WYKRES 1: TOP 10 najlepszych pojedynczych wyników
    """

    print("\n📊 Generowanie wykresu 1: TOP 10 najlepszych wyników...")

    top10 = best_runs_df.nsmallest(10, 'best_distance').copy()
    top10['label'] = top10['run'] + ': ' + top10['experiment_name']

    fig, ax1 = plt.subplots(figsize=(16, 10))

    colors_bars = [COLORS['extended_palette'][i % len(COLORS['extended_palette'])]
                   for i in range(len(top10))]

    # Oś 1 (lewa): best_distance
    bars = ax1.barh(range(len(top10)), top10['best_distance'],
                    color=colors_bars, alpha=0.8, edgecolor='white', linewidth=2)

    for i, (idx, row) in enumerate(top10.iterrows()):
        ax1.text(row['best_distance'] + 5, i, f"{row['best_distance']:.2f}",
                 va='center', fontweight='bold', fontsize=13)

    ax1.set_yticks(range(len(top10)))
    ax1.set_yticklabels(top10['label'], fontsize=12)
    ax1.set_xlabel('Długość najlepszej trasy', fontsize=16, fontweight='bold', color=COLORS['primary'])
    ax1.tick_params(axis='x', labelcolor=COLORS['primary'])
    ax1.grid(axis='x', alpha=0.3, linewidth=1)
    ax1.invert_yaxis()

    # Linia optimum
    optimal = 386.33
    ax1.axvline(optimal, color='green', linestyle='--', linewidth=3,
                alpha=0.7, label=f'Optimum (Held-Karp): {optimal}')

    # Oś 2 (prawa): generations_completed
    ax2 = ax1.twiny()

    ax2.plot(top10['generations_completed'].values, range(len(top10)),
             color=COLORS['accent'], marker='o', markersize=10, linewidth=3,
             linestyle='-', alpha=0.9, label='Liczba generacji', zorder=10)

    for i, (idx, row) in enumerate(top10.iterrows()):
        ax2.text(row['generations_completed'] + 5, i, f"{int(row['generations_completed'])}",
                 va='center', fontweight='bold', fontsize=11,
                 color=COLORS['accent'],
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor=COLORS['accent'], linewidth=2, alpha=0.9))

    ax2.set_xlabel('Liczba generacji', fontsize=16, fontweight='bold', color=COLORS['accent'])
    ax2.tick_params(axis='x', labelcolor=COLORS['accent'])
    ax2.grid(axis='x', alpha=0.2, linewidth=1, linestyle=':')

    ax1.set_title('🏆 TOP 10 Najlepszych Pojedynczych Wyników\n(z wszystkich eksperymentów × runs)',
                  fontsize=22, fontweight='bold', pad=60)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=13)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'chart1_top10_best_results.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Zapisano: chart1_top10_best_results.png")
    plt.close()


def chart2_stability_vs_quality(full_df):
    """
    WYKRES 2: Scatter plot - Stabilność vs Jakość
    """

    print("\n📊 Generowanie wykresu 2: Stabilność vs Jakość...")

    fig, ax = plt.subplots(figsize=(16, 12))

    unique_experiments = full_df['experiment_name'].unique()
    color_map = {exp: COLORS['extended_palette'][i % len(COLORS['extended_palette'])]
                 for i, exp in enumerate(unique_experiments)}

    colors = [color_map[exp] for exp in full_df['experiment_name']]

    scatter = ax.scatter(full_df['best_distance_mean'],
                         full_df['cv_distance'],
                         s=200, alpha=0.7, c=colors,
                         edgecolors='white', linewidth=2)

    for idx, row in full_df.iterrows():
        label = f"{row['run']}\n{row['experiment_name']}"
        ax.annotate(label,
                    (row['best_distance_mean'], row['cv_distance']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              alpha=0.7, edgecolor='gray', linewidth=1))

    median_mean = full_df['best_distance_mean'].median()
    median_cv = full_df['cv_distance'].median()

    ax.axhline(median_cv, color=COLORS['secondary'], linestyle='--',
               alpha=0.5, linewidth=2, label=f'Mediana CV: {median_cv:.1f}%')
    ax.axvline(median_mean, color=COLORS['primary'], linestyle='--',
               alpha=0.5, linewidth=2, label=f'Mediana mean: {median_mean:.1f}')

    ax.axhline(5, color='green', linestyle=':', alpha=0.3, linewidth=2)
    ax.axhline(10, color='orange', linestyle=':', alpha=0.3, linewidth=2)
    ax.text(ax.get_xlim()[0] + 5, 2.5, 'Bardzo stabilne (CV<5%)',
            fontsize=11, color='green', fontweight='bold')
    ax.text(ax.get_xlim()[0] + 5, 7.5, 'Umiarkowanie stabilne (CV 5-10%)',
            fontsize=11, color='orange', fontweight='bold')
    ax.text(ax.get_xlim()[0] + 5, 12, 'Niestabilne (CV>10%)',
            fontsize=11, color='red', fontweight='bold')

    ax.set_xlabel('Średnia długość trasy (niższa = lepiej)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Współczynnik zmienności CV% (niższy = stabilniej)', fontsize=18, fontweight='bold')
    ax.set_title('📊 Stabilność vs Jakość Eksperymentów\nKażdy punkt = (run × eksperyment)',
                 fontsize=22, fontweight='bold', pad=20)
    ax.grid(alpha=0.3, linewidth=1)
    ax.legend(fontsize=14, loc='upper right')

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'chart2_stability_vs_quality.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Zapisano: chart2_stability_vs_quality.png")
    plt.close()


def chart3_three_perspectives(full_df):
    """
    WYKRES 3: Trzy perspektywy - TOP 5 dla każdej
    """

    print("\n📊 Generowanie wykresu 3: Trzy perspektywy...")

    top_quality = full_df.nsmallest(5, 'best_distance_mean')
    top_stable = full_df.nsmallest(5, 'cv_distance')
    top_compromise = full_df.nsmallest(5, 'compromise_score')

    fig, axes = plt.subplots(1, 3, figsize=(24, 10))

    def plot_ranking(ax, data, metric, title, xlabel, color):
        data = data.copy()
        data['label'] = data['run'] + '\n' + data['experiment_name']

        bars = ax.barh(range(len(data)), data[metric],
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
                 '🏆 TOP 5\nNajlepsze Wyniki',
                 'Średnia długość trasy',
                 COLORS['primary'])

    plot_ranking(axes[1], top_stable, 'cv_distance',
                 '✅ TOP 5\nNajbardziej Stabilne',
                 'CV (%)',
                 COLORS['accent'])

    plot_ranking(axes[2], top_compromise, 'compromise_score',
                 '⚖️ TOP 5\nNajlepszy Kompromis',
                 'Wynik kompromisowy (niższy = lepiej)',
                 COLORS['secondary'])

    plt.suptitle('Trzy Perspektywy Analizy Eksperymentów',
                 fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'chart3_three_perspectives.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Zapisano: chart3_three_perspectives.png")
    plt.close()


def chart4_cv_distribution(full_df):
    """
    WYKRES 4: Rozkład CV
    """

    print("\n📊 Generowanie wykresu 4: Rozkład CV...")

    fig, ax = plt.subplots(figsize=(14, 10))

    ax.hist(full_df['cv_distance'], bins=20, alpha=0.6, color=COLORS['primary'],
            edgecolor='white', linewidth=2, label='Histogram')

    ax.axvline(5, color='green', linestyle='--', linewidth=3,
               label='Bardzo stabilne (CV=5%)')
    ax.axvline(10, color='orange', linestyle='--', linewidth=3,
               label='Umiarkowanie stabilne (CV=10%)')

    median_cv = full_df['cv_distance'].median()
    mean_cv = full_df['cv_distance'].mean()

    ax.axvline(median_cv, color='red', linestyle='-', linewidth=3,
               label=f'Mediana: {median_cv:.1f}%')
    ax.axvline(mean_cv, color='darkred', linestyle=':', linewidth=3,
               label=f'Średnia: {mean_cv:.1f}%')

    ax.set_xlabel('Współczynnik Zmienności CV (%)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Liczba eksperymentów', fontsize=18, fontweight='bold')
    ax.set_title('📊 Rozkład Stabilności Eksperymentów\n(Współczynnik Zmienności)',
                 fontsize=22, fontweight='bold', pad=20)
    ax.legend(fontsize=14, loc='upper right')
    ax.grid(alpha=0.3)

    very_stable = (full_df['cv_distance'] < 5).sum()
    moderate = ((full_df['cv_distance'] >= 5) & (full_df['cv_distance'] < 10)).sum()
    unstable = (full_df['cv_distance'] >= 10).sum()

    summary_text = f"""Podsumowanie:
Bardzo stabilne (CV<5%): {very_stable} ({very_stable / len(full_df) * 100:.1f}%)
Umiarkowanie (CV 5-10%): {moderate} ({moderate / len(full_df) * 100:.1f}%)
Niestabilne (CV>10%): {unstable} ({unstable / len(full_df) * 100:.1f}%)"""

    ax.text(0.98, 0.97, summary_text, transform=ax.transAxes,
            fontsize=13, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'chart4_cv_distribution.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Zapisano: chart4_cv_distribution.png")
    plt.close()


def chart5_parameter_comparison(full_df):
    """
    WYKRES 5: Porównanie parametrów
    """

    print("\n📊 Generowanie wykresu 5: Porównanie parametrów...")

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # 1. Crossover type - jakość
    if 'crossover_type' in full_df.columns:
        crossover_stats = full_df.groupby('crossover_type').agg({
            'best_distance_mean': ['mean', 'std'],
            'cv_distance': 'mean'
        }).round(2)
        crossover_stats.columns = ['mean_distance', 'std_distance', 'mean_cv']

        ax = axes[0, 0]
        crossover_stats['mean_distance'].plot(kind='bar', ax=ax,
                                              color=COLORS['extended_palette'][:len(crossover_stats)],
                                              alpha=0.8, edgecolor='white', linewidth=2)
        ax.set_title('Typ Crossover - Średnia długość trasy', fontsize=16, fontweight='bold')
        ax.set_xlabel('Typ crossover', fontsize=14)
        ax.set_ylabel('Średnia długość trasy', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
    # 3. Crossover - stabilność
    if 'crossover_type' in full_df.columns and full_df['crossover_type'].notna().any():
        ax = axes[1, 0]
        crossover_stats['mean_cv'].plot(kind='bar', ax=ax,
                                        color=COLORS['accent'],
                                        alpha=0.8, edgecolor='white', linewidth=2)
        ax.set_title('Typ Crossover - Stabilność (CV)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Typ crossover', fontsize=14)
        ax.set_ylabel('Średni CV (%)', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(5, color='green', linestyle='--', alpha=0.5, label='Bardzo stabilne')
        ax.axhline(10, color='orange', linestyle='--', alpha=0.5, label='Umiarkowanie')
        ax.legend()

        for i, (idx, val) in enumerate(crossover_stats['mean_cv'].items()):
            ax.text(i, val + 0.3, f"{val:.1f}%", ha='center', fontweight='bold')
    else:
        axes[1, 0].text(0.5, 0.5, 'Brak danych o crossover_type',
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Typ Crossover - Stabilność (CV)', fontsize=16, fontweight='bold')

    # 4. Selection - stabilność
    if 'parent_selection_type' in full_df.columns and full_df['parent_selection_type'].notna().any():
        ax = axes[1, 1]
        selection_stats['mean_cv'].plot(kind='bar', ax=ax,
                                        color=COLORS['accent'],
                                        alpha=0.8, edgecolor='white', linewidth=2)
        ax.set_title('Typ Selekcji - Stabilność (CV)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Typ selekcji', fontsize=14)
        ax.set_ylabel('Średni CV (%)', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(5, color='green', linestyle='--', alpha=0.5, label='Bardzo stabilne')
        ax.axhline(10, color='orange', linestyle='--', alpha=0.5, label='Umiarkowanie')
        ax.legend()

        for i, (idx, val) in enumerate(selection_stats['mean_cv'].items()):
            ax.text(i, val + 0.3, f"{val:.1f}%", ha='center', fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'Brak danych o parent_selection_type',
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Typ Selekcji - Stabilność (CV)', fontsize=16, fontweight='bold')

    plt.suptitle('Wpływ Parametrów na Wyniki i Stabilność',
                 fontsize=24, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'chart5_parameter_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Zapisano: chart5_parameter_comparison.png")
    plt.close()


def chart6_experiment_comparison_boxplot():
    """
    WYKRES 6: Boxplot wszystkich eksperymentów (raw data)
    """

    print("\n📊 Generowanie wykresu 6: Boxplot eksperymentów (raw data)...")

    base_path = Path(EXPERIMENTS_DIR)
    result_files = sorted(base_path.glob('experiment_results_run_*.csv'))

    all_data = []
    for file in result_files:
        df = pd.read_csv(file)
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)

    print(f"   Wczytano {len(combined)} wierszy z wszystkich runs")

    fig, ax = plt.subplots(figsize=(18, 12))

    exp_order = combined.groupby('experiment_name')['best_distance'].median().sort_values().index

    data_for_box = [combined[combined['experiment_name'] == exp]['best_distance'].values
                    for exp in exp_order]

    colors = [COLORS['extended_palette'][i % len(COLORS['extended_palette'])]
              for i in range(len(exp_order))]

    bp = ax.boxplot(data_for_box, labels=exp_order, vert=False, patch_artist=True,
                    boxprops=dict(linewidth=2.5, edgecolor='black'),
                    whiskerprops=dict(linewidth=2.5, color='black'),
                    capprops=dict(linewidth=2.5, color='black'),
                    medianprops=dict(linewidth=4, color='white'))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, exp in enumerate(exp_order):
        count = (combined['experiment_name'] == exp).sum()
        median = combined[combined['experiment_name'] == exp]['best_distance'].median()
        ax.text(median, i + 1, f' n={count}', va='center', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['primary'],
                          edgecolor='white', linewidth=2, alpha=0.9),
                color='white')

    ax.set_xlabel('Długość trasy (best_distance)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Eksperyment', fontsize=18, fontweight='bold')
    ax.set_title('📦 Porównanie Wszystkich Eksperymentów\n(Wszystkie próby ze wszystkich runs)',
                 fontsize=22, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linewidth=1)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'chart6_experiment_comparison_boxplot.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Zapisano: chart6_experiment_comparison_boxplot.png")
    plt.close()


def chart7_new_experiments_spotlight(full_df):
    """
    WYKRES 7: Spotlight na nowe eksperymenty
    """

    print("\n📊 Generowanie wykresu 7: Spotlight na nowe eksperymenty...")

    # Sprawdź czy mamy wymagane kolumny
    if 'crossover_type' not in full_df.columns or 'K_tournament' not in full_df.columns:
        print("   ⚠️  Brak kolumn crossover_type lub K_tournament - pomijam wykres")
        return

    # Zidentyfikuj nowe eksperymenty (OX crossover lub K=5)
    new_exp_mask = (
            (full_df['crossover_type'].str.lower() == 'ox') |
            (full_df['K_tournament'] == 5)
    )

    new_experiments = full_df[new_exp_mask].copy()

    if len(new_experiments) == 0:
        print("   ⚠️  Nie znaleziono nowych eksperymentów (OX lub K=5) - pomijam wykres")
        return

    baseline = full_df[full_df['experiment_name'] == 'baseline']

    if len(baseline) == 0:
        print("   ⚠️  Nie znaleziono eksperymentu 'baseline' - pomijam wykres")
        return

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Wykres 1: Porównanie średnich
    ax = axes[0]

    comparison_data = pd.DataFrame({
        'Baseline (średnia)': [baseline['best_distance_mean'].mean()],
        'OX Crossover': [
            new_experiments[new_experiments['crossover_type'].str.lower() == 'ox']['best_distance_mean'].mean()
            if (new_experiments['crossover_type'].str.lower() == 'ox').any() else 0],
        'K=5 Tournament': [new_experiments[new_experiments['K_tournament'] == 5]['best_distance_mean'].mean()
                           if (new_experiments['K_tournament'] == 5).any() else 0]
    })

    comparison_data = comparison_data.loc[:, (comparison_data != 0).any(axis=0)]

    bars = ax.bar(range(len(comparison_data.columns)), comparison_data.iloc[0],
                  color=[COLORS['primary'], COLORS['accent'], COLORS['secondary']][:len(comparison_data.columns)],
                  alpha=0.8, edgecolor='white', linewidth=3)

    ax.set_xticks(range(len(comparison_data.columns)))
    ax.set_xticklabels(comparison_data.columns, fontsize=14, fontweight='bold')
    ax.set_ylabel('Średnia długość trasy', fontsize=16, fontweight='bold')
    ax.set_title('🆕 Nowe Eksperymenty vs Baseline\nŚrednia długość trasy',
                 fontsize=18, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3)

    for i, (col, val) in enumerate(comparison_data.iloc[0].items()):
        ax.text(i, val + 5, f"{val:.1f}", ha='center', fontweight='bold', fontsize=14)

        if col != 'Baseline (średnia)':
            diff_pct = ((val - comparison_data.iloc[0]['Baseline (średnia)']) /
                        comparison_data.iloc[0]['Baseline (średnia)'] * 100)
            color = 'green' if diff_pct < 0 else 'red'
            ax.text(i, val - 15, f"{diff_pct:+.1f}%", ha='center',
                    fontweight='bold', fontsize=12, color=color)

    # Wykres 2: Porównanie stabilności
    ax = axes[1]

    cv_data = pd.DataFrame({
        'Baseline (średnia)': [baseline['cv_distance'].mean()],
        'OX Crossover': [new_experiments[new_experiments['crossover_type'].str.lower() == 'ox']['cv_distance'].mean()
                         if (new_experiments['crossover_type'].str.lower() == 'ox').any() else 0],
        'K=5 Tournament': [new_experiments[new_experiments['K_tournament'] == 5]['cv_distance'].mean()
                           if (new_experiments['K_tournament'] == 5).any() else 0]
    })

    cv_data = cv_data.loc[:, (cv_data != 0).any(axis=0)]

    bars = ax.bar(range(len(cv_data.columns)), cv_data.iloc[0],
                  color=[COLORS['primary'], COLORS['accent'], COLORS['secondary']][:len(cv_data.columns)],
                  alpha=0.8, edgecolor='white', linewidth=3)

    ax.set_xticks(range(len(cv_data.columns)))
    ax.set_xticklabels(cv_data.columns, fontsize=14, fontweight='bold')
    ax.set_ylabel('Współczynnik Zmienności CV (%)', fontsize=16, fontweight='bold')
    ax.set_title('🆕 Nowe Eksperymenty vs Baseline\nStabilność (CV)',
                 fontsize=18, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(5, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Bardzo stabilne')
    ax.axhline(10, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Umiarkowanie')
    ax.legend(fontsize=12)

    for i, (col, val) in enumerate(cv_data.iloc[0].items()):
        ax.text(i, val + 0.3, f"{val:.1f}%", ha='center', fontweight='bold', fontsize=14)

        if col != 'Baseline (średnia)':
            diff_pct = ((val - cv_data.iloc[0]['Baseline (średnia)']) /
                        cv_data.iloc[0]['Baseline (średnia)'] * 100)
            color = 'green' if diff_pct < 0 else 'red'
            ax.text(i, val - 0.5, f"{diff_pct:+.1f}%", ha='center',
                    fontweight='bold', fontsize=12, color=color)

    plt.suptitle('Analiza Nowych Eksperymentów',
                 fontsize=24, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'chart7_new_experiments_spotlight.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Zapisano: chart7_new_experiments_spotlight.png")
    plt.close()


def chart8_consistency_heatmap(full_df):
    """
    WYKRES 8: Heatmapa konsystencji
    """

    print("\n📊 Generowanie wykresu 8: Heatmapa konsystencji...")

    pivot = full_df.pivot(index='experiment_name', columns='run', values='best_distance_mean')

    pivot = pivot[sorted(pivot.columns, key=lambda x: int(x.replace('run_', '')))]
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(20, 12))

    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Średnia długość trasy', fontsize=16, fontweight='bold')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=12)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=12)

    ax.set_xlabel('Run', fontsize=18, fontweight='bold')
    ax.set_ylabel('Eksperyment', fontsize=18, fontweight='bold')
    ax.set_title('🔥 Heatmapa Konsystencji Eksperymentów\n(Ciemniejszy = lepszy wynik)',
                 fontsize=22, fontweight='bold', pad=20)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            if pd.notna(pivot.values[i, j]):
                text = ax.text(j, i, f'{pivot.values[i, j]:.0f}',
                               ha="center", va="center", color="black", fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'chart8_consistency_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Zapisano: chart8_consistency_heatmap.png")
    plt.close()


# ============================================================================
# KROK 4: RAPORT TEKSTOWY
# ============================================================================

def generate_summary_report(best_runs_df, full_df):
    """
    Generuje tekstowy raport podsumowujący
    """

    print("\n" + "=" * 80)
    print("KROK 4: GENEROWANIE RAPORTU PODSUMOWUJĄCEGO")
    print("=" * 80 + "\n")

    report = []
    report.append("=" * 80)
    report.append("RAPORT PODSUMOWUJĄCY - ANALIZA EKSPERYMENTÓW")
    report.append("=" * 80)
    report.append("")

    # Podstawowe statystyki
    report.append("PODSTAWOWE STATYSTYKI:")
    report.append(f"  Liczba runs: {full_df['run'].nunique()}")
    report.append(f"  Liczba unikalnych eksperymentów: {full_df['experiment_name'].nunique()}")
    report.append(f"  Łączna liczba (run × experiment): {len(full_df)}")
    report.append("")

    # Najlepszy wynik
    best_overall = best_runs_df.loc[best_runs_df['best_distance'].idxmin()]
    report.append("🥇 ABSOLUTNIE NAJLEPSZY WYNIK:")
    report.append(f"  Run: {best_overall['run']}")
    report.append(f"  Eksperyment: {best_overall['experiment_name']}")
    if 'description' in best_overall.index:
        report.append(f"  Opis: {best_overall['description']}")
    report.append(f"  Best distance: {best_overall['best_distance']:.2f}")
    report.append(f"  Parametry:")
    report.append(f"    - Population size: {best_overall['population_size']}")
    report.append(f"    - Mutation: {best_overall['mutation_percent_genes']}%")
    report.append(f"    - Crossover: {best_overall['crossover_type']}")
    report.append(f"    - Selection: {best_overall['parent_selection_type']}")
    if pd.notna(best_overall.get('K_tournament')):
        report.append(f"    - K tournament: {int(best_overall['K_tournament'])}")
    report.append(f"    - Generations: {int(best_overall['generations_completed'])}")
    report.append("")

    # TOP 3 perspektywy
    report.append("🏆 TOP 3 - NAJLEPSZE ŚREDNIE WYNIKI:")
    top_quality = full_df.nsmallest(3, 'best_distance_mean')
    for idx, row in enumerate(top_quality.itertuples(), 1):
        report.append(f"  {idx}. {row.run} - {row.experiment_name}")
        report.append(f"     Średnia: {row.best_distance_mean:.2f}, CV: {row.cv_distance:.2f}%")
    report.append("")

    report.append("✅ TOP 3 - NAJBARDZIEJ STABILNE (najmniejszy CV):")
    top_stable = full_df.nsmallest(3, 'cv_distance')
    for idx, row in enumerate(top_stable.itertuples(), 1):
        report.append(f"  {idx}. {row.run} - {row.experiment_name}")
        report.append(f"     CV: {row.cv_distance:.2f}%, Średnia: {row.best_distance_mean:.2f}")
    report.append("")

    report.append("⚖️  TOP 3 - NAJLEPSZY KOMPROMIS:")
    top_compromise = full_df.nsmallest(3, 'compromise_score')
    for idx, row in enumerate(top_compromise.itertuples(), 1):
        report.append(f"  {idx}. {row.run} - {row.experiment_name}")
        report.append(
            f"     Średnia: {row.best_distance_mean:.2f}, CV: {row.cv_distance:.2f}%, Score: {row.compromise_score:.3f}")
    report.append("")

    # Analiza stabilności
    report.append("📊 ANALIZA STABILNOŚCI:")
    very_stable = (full_df['cv_distance'] < 5).sum()
    moderate = ((full_df['cv_distance'] >= 5) & (full_df['cv_distance'] < 10)).sum()
    unstable = (full_df['cv_distance'] >= 10).sum()

    report.append(f"  Bardzo stabilne (CV<5%): {very_stable} ({very_stable / len(full_df) * 100:.1f}%)")
    report.append(f"  Umiarkowanie (CV 5-10%): {moderate} ({moderate / len(full_df) * 100:.1f}%)")
    report.append(f"  Niestabilne (CV>10%): {unstable} ({unstable / len(full_df) * 100:.1f}%)")
    report.append(f"  Średni CV: {full_df['cv_distance'].mean():.2f}%")
    report.append(f"  Mediana CV: {full_df['cv_distance'].median():.2f}%")
    report.append("")

    # Porównanie parametrów
    if 'crossover_type' in full_df.columns:
        report.append("🔀 PORÓWNANIE TYPÓW CROSSOVER:")
        crossover_stats = full_df.groupby('crossover_type').agg({
            'best_distance_mean': 'mean',
            'cv_distance': 'mean'
        }).round(2)
        for ctype, row in crossover_stats.iterrows():
            report.append(f"  {ctype}: mean={row['best_distance_mean']:.2f}, CV={row['cv_distance']:.2f}%")
        report.append("")

    if 'parent_selection_type' in full_df.columns:
        report.append("👥 PORÓWNANIE TYPÓW SELEKCJI:")
        selection_stats = full_df.groupby('parent_selection_type').agg({
            'best_distance_mean': 'mean',
            'cv_distance': 'mean'
        }).round(2)
        for stype, row in selection_stats.iterrows():
            report.append(f"  {stype}: mean={row['best_distance_mean']:.2f}, CV={row['cv_distance']:.2f}%")
        report.append("")

    # Wnioski
    report.append("=" * 80)
    report.append("💡 WNIOSKI DLA STUDENTÓW:")
    report.append("=" * 80)
    report.append("")
    report.append("1. ALGORYTMY GENETYCZNE SĄ STOCHASTYCZNE")
    report.append("   - Te same parametry dają różne wyniki w różnych uruchomieniach")
    report.append("   - ZAWSZE testuj wiele razy i raportuj statystyki (mean, std, CV)")
    report.append("")
    report.append("2. STABILNOŚĆ VS JAKOŚĆ")
    avg_cv = full_df['cv_distance'].mean()
    if avg_cv < 5:
        report.append("   - Eksperymenty są BARDZO STABILNE (średni CV < 5%)")
    elif avg_cv < 10:
        report.append("   - Eksperymenty są UMIARKOWANIE STABILNE (średni CV 5-10%)")
    else:
        report.append("   - Eksperymenty są NIESTABILNE (średni CV > 10%)")
        report.append("   - Rozważ zwiększenie liczby generacji lub populacji")
    report.append("")
    report.append("3. WYBÓR KONFIGURACJI")
    report.append("   - Dla badań: wybierz konfigurację z najlepszą średnią")
    report.append("   - Dla produkcji: wybierz konfigurację z najlepszym kompromisem")
    report.append("   - Unikaj konfiguracji z CV > 10% (zbyt niestabilne)")
    report.append("")
    report.append("4. WSPÓŁCZYNNIK ZMIENNOŚCI (CV)")
    report.append("   - CV = (std / mean) × 100%")
    report.append("   - Pozwala porównywać stabilność między eksperymentami")
    report.append("   - CV < 5% = doskonała stabilność")
    report.append("   - CV > 10% = wymaga więcej prób lub lepszych parametrów")
    report.append("")

    report.append("=" * 80)
    report.append("KONIEC RAPORTU")
    report.append("=" * 80)

    # Zapisz raport
    report_text = "\n".join(report)

    report_file = Path(OUTPUT_DIR) / 'summary_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"✅ Zapisano raport do: summary_report.txt")
    print("\n" + report_text)


# ============================================================================
# MAIN - URUCHOMIENIE CAŁEGO PIPELINE
# ============================================================================

def main():
    """
    Główna funkcja - uruchamia cały pipeline analizy
    """

    print("\n" + "=" * 80)
    print("ZUNIFIKOWANA ANALIZA EKSPERYMENTÓW - ALGORYTMY GENETYCZNE TSP")
    print("=" * 80)
    print(f"\nKatalog z danymi: {EXPERIMENTS_DIR}/")
    print(f"Katalog wynikowy: {OUTPUT_DIR}/")
    print("=" * 80 + "\n")

    # Utwórz foldery wynikowe
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Utworzono strukturę folderów:")
    print(f"  - {DATA_DIR}/")
    print(f"  - {CHARTS_DIR}/")

    # ========================================================================
    # KROK 1: Zbieranie danych
    # ========================================================================

    result = collect_all_experiment_data()

    if result[0] is None:
        print("\n❌ Przerwano analizę - brak danych")
        return

    best_runs_df, stability_df, full_df = result

    # Zapisz dane do CSV
    best_runs_df.to_csv(DATA_DIR / 'best_runs_per_experiment.csv', index=False)
    stability_df.to_csv(DATA_DIR / 'stability_per_run.csv', index=False)
    full_df.to_csv(DATA_DIR / 'full_experiment_data.csv', index=False)

    print(f"\n✅ Zapisano pliki CSV:")
    print(f"  - {DATA_DIR / 'best_runs_per_experiment.csv'}")
    print(f"  - {DATA_DIR / 'stability_per_run.csv'}")
    print(f"  - {DATA_DIR / 'full_experiment_data.csv'}")

    # ========================================================================
    # KROK 2: Analiza i rankingi
    # ========================================================================

    analyze_rankings(full_df)

    # ========================================================================
    # KROK 3: Generowanie wykresów
    # ========================================================================

    print("\n" + "=" * 80)
    print("KROK 3: GENEROWANIE WYKRESÓW")
    print("=" * 80)

    chart1_top10_best_results(best_runs_df)
    chart2_stability_vs_quality(full_df)
    chart3_three_perspectives(full_df)
    chart4_cv_distribution(full_df)
    chart5_parameter_comparison(full_df)
    chart6_experiment_comparison_boxplot()
    chart7_new_experiments_spotlight(full_df)
    chart8_consistency_heatmap(full_df)

    print(f"\n✅ Wszystkie wykresy zapisane w: {CHARTS_DIR}/")

    # ========================================================================
    # KROK 4: Raport tekstowy
    # ========================================================================

    generate_summary_report(best_runs_df, full_df)

    # ========================================================================
    # PODSUMOWANIE
    # ========================================================================

    print("\n" + "=" * 80)
    print("✅ ANALIZA ZAKOŃCZONA POMYŚLNIE!")
    print("=" * 80)
    print(f"\nWygenerowano:")
    print(f"  📁 3 pliki CSV w: {DATA_DIR}/")
    print(f"  📊 8 wykresów PNG w: {CHARTS_DIR}/")
    print(f"  📄 1 raport TXT w: {OUTPUT_DIR}/")
    print("\nLista plików:")
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
    print("    - chart6_experiment_comparison_boxplot.png")
    print("    - chart7_new_experiments_spotlight.png")
    print("    - chart8_consistency_heatmap.png")
    print("  Raport:")
    print("    - summary_report.txt")
    print("=" * 80)


if __name__ == "__main__":
    main()

    for i, (idx, val) in enumerate(crossover_stats['mean_distance'].items()):
        ax.text(i, val + 5, f"{val:.1f}", ha='center', fontweight='bold')

# 2. Selection type - jakość
if 'parent_selection_type' in full_df.columns:
    selection_stats = full_df.groupby('parent_selection_type').agg({
        'best_distance_mean': ['mean', 'std'],
        'cv_distance': 'mean'
    }).round(2)
    selection_stats.columns = ['mean_distance', 'std_distance', 'mean_cv']

    ax = axes[0, 1]
    selection_stats['mean_distance'].plot(kind='bar', ax=ax,
                                          color=COLORS['extended_palette'][:len(selection_stats)],
                                          alpha=0.8, edgecolor='white', linewidth=2)
    ax.set_title('Typ Selekcji - Średnia długość trasy', fontsize=16, fontweight='bold')
    ax.set_xlabel('Typ selekcji', fontsize=14)
    ax.set_ylabel('Średnia długość trasy', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    for i, (idx, val) in enumerate(selection_stats['mean_distance'].items()):
        ax.text(i, val + 5, f"{val:.1f}", ha='center', fontweight='bold')

# 3. Crossover - stabilność
if 'crossover_type' in full_df.columns:
    ax = axes[1, 0]
    crossover_stats['mean_cv'].plot(kind='bar', ax=ax,
                                    color=COLORS['accent'],
                                    alpha=0.8, edgecolor='white', linewidth=2)
    ax.set_title('Typ Crossover - Stabilność (CV)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Typ crossover', fontsize=14)
    ax.set_ylabel('Średni CV (%)', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)