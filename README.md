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

**📖 Szczegółowy przewodnik:**  
Zobacz [MULTI_RUN_GUIDE.md](MULTI_RUN_GUIDE.md) aby dowiedzieć się:
- Jak uruchomić wiele eksperymentów automatycznie (np. 30 razy)
- Jak wyłączyć wizualizacje dla szybszego działania
- Jak organizować wyniki po każdym batch'u
- Przykładowe komendy dla Linux/Mac/Windows

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

---

## 📂 Project Structure

```
project/
├── main.py                              # Run individual experiments
├── utils.py                             # Helper functions (TSP, GA, visualizations)
├── config.yaml                          # Experiment configuration
├── ga_tsp_analyses.py                   # Single run analysis
├── run_analyses.py                      # ⭐ Multi-run analysis
├── utils_unified_analyses.py            # ⭐ Multi-run analysis functions
├── experiments/                         # Data folder for multiple runs
│   ├── experiment_results_run_1.csv
│   ├── experiment_results_run_2.csv
│   └── ...
├── outputs/                             # Single run results
│   ├── convergence/
│   ├── routes/
│   ├── animations/
│   └── analysis/
└── analysis_results/                    # ⭐ Multi-run analysis results
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

## 🚀 Running the Program

### PART 1: Individual Experiments

**Run main experiments:**
```bash
python main.py
```
The script will automatically:
- Execute all experiments defined in `config.yaml`
- Generate visualizations in the `outputs/` folder
- Save results to `experiment_results.csv`

**Run single-run analysis:**
```bash
python ga_tsp_analyses.py
```
The script will generate:
- Experiment stability statistics
- Comparative charts (boxplot, scatter plot, histograms)
- Save analysis in `outputs/analysis/` folder

---

### PART 2: Multi-Run Analysis (Advanced)

**If you have multiple result files** (e.g., `experiment_results_run_1.csv`, `run_2.csv`, etc.) **in the `experiments/` folder:**

```bash
python run_analyses.py
```

The script will automatically:
- Collect data from all `experiment_results_run_*.csv` files
- Calculate stability statistics for each (run × experiment)
- Generate **7 advanced charts**:
  1. TOP 10 best individual results
  2. Scatter plot: Stability vs Quality
  3. Three perspectives TOP 5 (quality, stability, compromise)
  4. Coefficient of variation (CV) distribution
  5. Parameter impact (crossover, selection) on results
  6. Boxplot comparing all experiments
  7. Experiment consistency heatmap
- Save a text report with conclusions

**Results will be in:** `analysis_results/`

**📖 Detailed Guide:**  
See [MULTI_RUN_GUIDE.md](MULTI_RUN_GUIDE.md) to learn:
- How to run multiple experiments automatically (e.g., 30 times)
- How to disable visualizations for faster execution
- How to organize results after each batch
- Example commands for Linux/Mac/Windows

---

### Output Structure - Single Run
```
outputs/
├── convergence/        # Algorithm convergence plots
├── routes/            # Best route visualizations
├── animations/        # GIF animations of solution evolution
└── analysis/          # Statistics and comparative analysis
```

### Output Structure - Multiple Runs
```
analysis_results/
├── data/              # 3 CSV files with aggregated data
├── charts/            # 7 PNG charts for presentations
└── summary_report.txt # Text report with conclusions
```

---

## 📊 Quick Tips

- Modify `config.yaml` to adjust experiment parameters
- Each `main.py` run appends results to `experiment_results.csv` (accumulative)
- For multi-run analysis, place CSV files in `experiments/` folder with naming: `experiment_results_run_*.csv`
- Animations may take a few minutes to generate
- All visualizations are saved at 300 DPI for presentation quality
- The multi-run analysis generates publication-ready charts and statistical reports