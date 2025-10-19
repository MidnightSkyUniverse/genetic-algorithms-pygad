# TSP Optimization using Genetic Algorithms

## 🇵🇱 Polish Version

### Wymagania
- Python 3.8+
- pip (menedżer pakietów Python)

### Instalacja bibliotek

```bash
pip install numpy matplotlib seaborn pandas pygad pyyaml
```

---

## 📂 Struktura Projektu

```
projekt/
├── main.py                              # Uruchamianie pojedynczych eksperymentów
├── utils.py                             # Funkcje pomocnicze (TSP, GA, wizualizacje)
├── config.yaml                          # Konfiguracja eksperymentów
├── ga_tsp_analyses.py                   # Analiza pojedynczego run
├── run_analyses.py                      # ⭐ Analiza wielu runs
├── utils_unified_analyses.py            # ⭐ Funkcje do analizy wielu runs
├── experiments/                         # Folder z danymi z wielu runs
│   ├── experiment_results_run_1.csv
│   ├── experiment_results_run_2.csv
│   └── ...
├── outputs/                             # Wyniki pojedynczego run
│   ├── convergence/
│   ├── routes/
│   ├── animations/
│   └── analysis/
└── analysis_results/                    # ⭐ Wyniki analizy wielu runs
    ├── data/
    │   ├── best_runs_per_experiment.csv
    │   ├── stability_per_run.csv
    │   └── full_experiment_data.csv
    ├── charts/
    │   ├── chart1_top10_best_results.png
    │   ├── chart2_stability_vs_quality.png
    │   ├── chart3_three_perspectives.png
    │   ├── chart4_cv_distribution.png
    │   ├── chart5_parameter_comparison.png
    │   ├── chart6_boxplot_all.png
    │   └── chart7_heatmap.png
    └── summary_report.txt
```

---

## 🚀 Uruchomienie Programu

### CZĘŚĆ 1: Pojedyncze Eksperymenty

**Uruchom główne eksperymenty:**
```bash
python main.py
```
Skrypt automatycznie:
- Uruchomi wszystkie eksperymenty zdefiniowane w `config.yaml`
- Wygeneruje wizualizacje w folderze `outputs/`
- Zapisze wyniki do `experiment_results.csv`

**Uruchom analizę pojedynczego run:**
```bash
python ga_tsp_analyses.py
```
Skrypt wygeneruje:
- Statystyki stabilności eksperymentów
- Wykresy porównawcze (boxplot, scatter plot, histogramy)
- Zapisze analizę w folderze `outputs/analysis/`

---

### CZĘŚĆ 2: Analiza Wielu Runs (Advanced)

**Jeśli masz wiele plików wyników** (np. `experiment_results_run_1.csv`, `run_2.csv`, etc.) **w folderze `experiments/`:**

```bash
python run_analyses.py
```

Skrypt automatycznie:
- Zbierze dane ze wszystkich plików `experiment_results_run_*.csv`
- Obliczy statystyki stabilności dla każdego (run × eksperyment)
- Wygeneruje **7 zaawansowanych wykresów**:
  1. TOP 10 najlepszych pojedynczych wyników
  2. Scatter plot: Stabilność vs Jakość
  3. Trzy perspektywy TOP 5 (jakość, stabilność, kompromis)
  4. Rozkład współczynnika zmienności (CV)
  5. Wpływ parametrów (crossover, selection) na wyniki
  6. Boxplot porównujący wszystkie eksperymenty
  7. Heatmapa konsystencji eksperymentów
- Zapisze raport tekstowy z wnioskami

**Wyniki znajdziesz w:** `analysis_results/`

---

### Struktura Wyników - Pojedynczy Run
```
outputs/
├── convergence/        # Wykresy zbieżności algorytmu
├── routes/            # Wizualizacje najlepszych tras
├── animations/        # Animacje GIF ewolucji rozwiązań
└── analysis/          # Statystyki i analizy porównawcze
```

### Struktura Wyników - Wiele Runs
```
analysis_results/
├── data/              # 3 pliki CSV z agregatami danych
├── charts/            # 7 wykresów PNG do prezentacji
└── summary_report.txt # Raport tekstowy z wnioskami
```

---

## 🇬🇧 English Version

### Requirements
- Python 3.8+
- pip (Python package manager)

### Library Installation

```bash
pip install numpy matplotlib seaborn pandas pygad pyyaml
```

### Running the Program

1. **Run main experiments:**
```bash
python main.py
```
The script will automatically:
- Execute all experiments defined in `config.yaml`
- Generate visualizations in the `outputs/` folder
- Save results to `experiment_results.csv`

2. **Run results analysis:**
```bash
python ga_tsp_analyses.py
```
The script will generate:
- Experiment stability statistics
- Comparative charts (boxplot, scatter plot, histograms)
- Save analysis in `outputs/analysis/` folder

### Output Structure
```
outputs/
├── convergence/        # Algorithm convergence plots
├── routes/            # Best route visualizations
├── animations/        # GIF animations of solution evolution
└── analysis/          # Statistics and comparative analysis
```

---

## 📊 Quick Tips

- Modify `config.yaml` to adjust experiment parameters
- Each run appends results to `experiment_results.csv` (accumulative)
- Animations may take a few minutes to generate
- All visualizations are saved at 300 DPI for presentation quality