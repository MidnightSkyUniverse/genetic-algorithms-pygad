import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Konfiguracja
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
sns.set_palette("husl")

# Wczytanie danych
df = pd.read_csv('experiment_results.csv')

# Utworzenie folderu na wyniki
output_dir = Path('outputs/analysis')
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("ANALIZA STABILNOŚCI EKSPERYMENTÓW - ALGORYTMY GENETYCZNE TSP")
print("=" * 80)

# ============================================================================
# 1. STATYSTYKI PODSTAWOWE DLA KAŻDEGO EKSPERYMENTU
# ============================================================================

print("\n" + "=" * 80)
print("1. STABILNOŚĆ KAŻDEGO EKSPERYMENTU")
print("=" * 80)

stability_stats = df.groupby('experiment_name').agg({
    'best_distance': ['count', 'mean', 'std', 'min', 'max'],
    'generations_completed': ['mean', 'std']
}).round(2)

stability_stats.columns = ['_'.join(col).strip() for col in stability_stats.columns]
stability_stats['cv_distance'] = (stability_stats['best_distance_std'] /
                                  stability_stats['best_distance_mean'] * 100).round(2)

# Sortowanie po współczynniku zmienności (od najbardziej stabilnego)
stability_stats = stability_stats.sort_values('cv_distance')

print("\nWspółczynnik Zmienności (CV) - im niższy, tym bardziej stabilny eksperyment:")
print(stability_stats[['best_distance_count', 'best_distance_mean', 'best_distance_std',
                       'best_distance_min', 'best_distance_max', 'cv_distance']])

# Zapisz do CSV
stability_stats.to_csv(output_dir / 'stability_summary.csv')
print(f"\n✓ Zapisano szczegółowe statystyki do: {output_dir / 'stability_summary.csv'}")

# ============================================================================
# 2. ANALIZA METOD SELEKCJI
# ============================================================================

print("\n" + "=" * 80)
print("2. PORÓWNANIE METOD SELEKCJI RODZICÓW")
print("=" * 80)

selection_stats = df.groupby('parent_selection_type').agg({
    'best_distance': ['count', 'mean', 'std', 'min', 'max']
}).round(2)

selection_stats.columns = ['_'.join(col).strip() for col in selection_stats.columns]
selection_stats['cv'] = (selection_stats['best_distance_std'] /
                         selection_stats['best_distance_mean'] * 100).round(2)

print("\nMetody selekcji (sortowane po średniej jakości):")
print(selection_stats.sort_values('best_distance_mean'))

# ============================================================================
# 3. WIZUALIZACJE
# ============================================================================

print("\n" + "=" * 80)
print("3. GENEROWANIE WIZUALIZACJI")
print("=" * 80)

# 3.1 BOXPLOT - Porównanie wszystkich eksperymentów
fig, ax = plt.subplots(figsize=(16, 10))
df_sorted = df.sort_values('best_distance', ascending=False)
exp_order = df.groupby('experiment_name')['best_distance'].median().sort_values().index

sns.boxplot(data=df, y='experiment_name', x='best_distance',
            order=exp_order, ax=ax)
ax.set_xlabel('Długość trasy (best_distance)', fontsize=12, fontweight='bold')
ax.set_ylabel('Eksperyment', fontsize=12, fontweight='bold')
ax.set_title('Porównanie stabilności eksperymentów - Boxplot\n(sortowane po medianie)',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Dodaj adnotacje z liczbą prób
for i, exp in enumerate(exp_order):
    count = df[df['experiment_name'] == exp].shape[0]
    median = df[df['experiment_name'] == exp]['best_distance'].median()
    ax.text(median, i, f' n={count}', va='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig(output_dir / 'boxplot_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Zapisano: {output_dir / 'boxplot_comparison.png'}")
plt.close()

# 3.2 SCATTER PLOT - Stabilność vs Jakość
fig, ax = plt.subplots(figsize=(12, 8))

exp_summary = df.groupby('experiment_name').agg({
    'best_distance': ['mean', 'std', 'count']
}).reset_index()
exp_summary.columns = ['experiment_name', 'mean_distance', 'std_distance', 'count']

scatter = ax.scatter(exp_summary['mean_distance'], exp_summary['std_distance'],
                     s=exp_summary['count'] * 50, alpha=0.6, c=range(len(exp_summary)),
                     cmap='viridis', edgecolors='black', linewidth=1.5)

# Adnotacje
for idx, row in exp_summary.iterrows():
    ax.annotate(row['experiment_name'],
                (row['mean_distance'], row['std_distance']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax.set_xlabel('Średnia długość trasy (niższa = lepiej)', fontsize=12, fontweight='bold')
ax.set_ylabel('Odchylenie standardowe (niższe = stabilniej)', fontsize=12, fontweight='bold')
ax.set_title('Stabilność vs Jakość eksperymentów\n(rozmiar punktu = liczba uruchomień)',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(alpha=0.3)

# Linie pomocnicze
ax.axhline(exp_summary['std_distance'].mean(), color='red', linestyle='--',
           alpha=0.5, label=f'Średnie std: {exp_summary["std_distance"].mean():.1f}')
ax.axvline(exp_summary['mean_distance'].mean(), color='blue', linestyle='--',
           alpha=0.5, label=f'Średnia mean: {exp_summary["mean_distance"].mean():.1f}')
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / 'stability_vs_quality.png', dpi=300, bbox_inches='tight')
print(f"✓ Zapisano: {output_dir / 'stability_vs_quality.png'}")
plt.close()

# 3.3 HISTOGRAMY dla każdego eksperymentu
experiments = df['experiment_name'].unique()
n_exp = len(experiments)
n_cols = 3
n_rows = (n_exp + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
axes = axes.flatten() if n_exp > 1 else [axes]

for idx, exp in enumerate(sorted(experiments)):
    exp_data = df[df['experiment_name'] == exp]['best_distance']

    axes[idx].hist(exp_data, bins=min(20, len(exp_data)),
                   alpha=0.7, color='steelblue', edgecolor='black')
    axes[idx].axvline(exp_data.mean(), color='red', linestyle='--',
                      linewidth=2, label=f'Średnia: {exp_data.mean():.1f}')
    axes[idx].axvline(exp_data.median(), color='green', linestyle='--',
                      linewidth=2, label=f'Mediana: {exp_data.median():.1f}')

    axes[idx].set_title(f'{exp}\n(n={len(exp_data)}, std={exp_data.std():.1f})',
                        fontweight='bold')
    axes[idx].set_xlabel('Długość trasy')
    axes[idx].set_ylabel('Częstość')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(alpha=0.3)

# Ukryj puste subploty
for idx in range(n_exp, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Rozkład wyników dla każdego eksperymentu',
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(output_dir / 'histograms_all.png', dpi=300, bbox_inches='tight')
print(f"✓ Zapisano: {output_dir / 'histograms_all.png'}")
plt.close()


# ============================================================================
# 4. WNIOSKI I REKOMENDACJE
# ============================================================================

print("\n" + "=" * 80)
print("4. WNIOSKI I REKOMENDACJE")
print("=" * 80)

# Najlepsze konfiguracje
best_mean = stability_stats.nsmallest(3, 'best_distance_mean')
most_stable = stability_stats.nsmallest(3, 'cv_distance')
least_stable = stability_stats.nlargest(3, 'cv_distance')

print("\n🏆 TOP 3 - NAJLEPSZE ŚREDNIE WYNIKI (najmniejsza trasa):")
for idx, (exp_name, row) in enumerate(best_mean.iterrows(), 1):
    print(f"{idx}. {exp_name}: {row['best_distance_mean']:.2f} (±{row['best_distance_std']:.2f})")

print("\n✅ TOP 3 - NAJSTABILNIEJSZE (najmniejszy CV):")
for idx, (exp_name, row) in enumerate(most_stable.iterrows(), 1):
    print(f"{idx}. {exp_name}: CV={row['cv_distance']:.2f}% (mean={row['best_distance_mean']:.2f})")

print("\n⚠️  TOP 3 - NAJBARDZIEJ NIESTABILNE (największy CV):")
for idx, (exp_name, row) in enumerate(least_stable.iterrows(), 1):
    print(f"{idx}. {exp_name}: CV={row['cv_distance']:.2f}% (mean={row['best_distance_mean']:.2f})")

# Diagnostyka niestabilności
print("\n" + "=" * 80)
print("DIAGNOSTYKA NIESTABILNOŚCI")
print("=" * 80)

# Sprawdź czy niektóre eksperymenty mają mało prób
low_samples = stability_stats[stability_stats['best_distance_count'] < 5]
if len(low_samples) > 0:
    print(f"\n⚠️  Eksperymenty z małą liczbą prób (< 5):")
    for exp_name, row in low_samples.iterrows():
        print(f"   - {exp_name}: tylko {int(row['best_distance_count'])} prób")
else:
    print("\n✓ Wszystkie eksperymenty mają wystarczającą liczbę prób")

# Sprawdź czy są duże różnice w liczbie generacji
gen_stats = df.groupby('experiment_name')['generations_completed'].agg(['mean', 'std', 'min', 'max'])
high_gen_variance = gen_stats[gen_stats['std'] > gen_stats['mean'] * 0.3]
if len(high_gen_variance) > 0:
    print(f"\n⚠️  Eksperymenty z dużą zmiennością liczby generacji:")
    for exp_name, row in high_gen_variance.iterrows():
        print(
            f"   - {exp_name}: {row['mean']:.0f} ±{row['std']:.0f} generacji (range: {int(row['min'])}-{int(row['max'])})")
else:
    print("\n✓ Liczba generacji jest stabilna we wszystkich eksperymentach")

# Rekomendacje
print("\n" + "=" * 80)
print("📋 REKOMENDACJE:")
print("=" * 80)

avg_cv = stability_stats['cv_distance'].mean()
if avg_cv > 10:
    print(f"\n1. Średni CV = {avg_cv:.1f}% - wyniki są BARDZO NIESTABILNE")
    print("   Rozważ:")
    print("   - Zwiększenie liczby uruchomień każdego eksperymentu (min. 30 powtórzeń)")
    print("   - Ustalenie stałego seed dla generatora liczb losowych")
    print("   - Wydłużenie czasu działania algorytmu (więcej generacji)")
elif avg_cv > 5:
    print(f"\n1. Średni CV = {avg_cv:.1f}% - wyniki są UMIARKOWANIE NIESTABILNE")
    print("   Rozważ:")
    print("   - Zwiększenie liczby powtórzeń do min. 20-30")
    print("   - Weryfikację czy parametry są odpowiednie dla problemu")
else:
    print(f"\n1. Średni CV = {avg_cv:.1f}% - wyniki są względnie STABILNE")

print(f"\n2. Aby poprawić porównywalność:")
print("   - Upewnij się, że każdy eksperyment ma taką samą liczbę uruchomień")
print("   - Używaj tego samego zestawu instancji TSP dla wszystkich eksperymentów")
print("   - Zapisuj więcej metryk (czas wykonania, różnorodność populacji)")

print("\n" + "=" * 80)
print("✓ ANALIZA ZAKOŃCZONA")
print(f"✓ Wszystkie wyniki zapisane w: {output_dir.absolute()}")
print("=" * 80)