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
├── main.py                              # Uruchamianie eksperymentów
├── utils.py                             # Funkcje pomocnicze (TSP, GA, wizualizacje)
├── config.yaml                          # Konfiguracja eksperymentów
├── ga_tsp_analyses.py                   # Analiza pojedynczego CSV (30 powtórzeń)
├── run_analyses.py                      # ⭐ Analiza wielu runs (porównawcza)
├── utils_unified_analyses.py            # ⭐ Funkcje do analizy wielu runs
├── held_karp.py                         # Algorytm Held-Karp (optymalna trasa)
├── MULTI_RUN_GUIDE.md                   # 📖 Szczegółowy przewodnik
├── experiments/                         # Folder z danymi z wielu runs
│   ├── experiment_results_run_1.csv
│   ├── experiment_results_run_2.csv
│   └── ...
├── outputs/                             # Wyniki bieżącego uruchomienia
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

### CZĘŚĆ 1: Podstawowe Uruchomienie

**Uruchom eksperymenty:**
```bash
python main.py
```
Skrypt automatycznie:
- Uruchomi wszystkie eksperymenty zdefiniowane w `config.yaml`
- Wygeneruje wizualizacje w folderze `outputs/`
- Doda wyniki do `experiment_results.csv` (append)

**💡 Uwaga:** Pojedyncze uruchomienie `main.py` generuje wykresy, ale **NIE MA SENSU analizować statystyk z jednego wyniku**. Do analizy potrzeba wielu powtórzeń (zobacz CZĘŚĆ 2).

---

### CZĘŚĆ 2: Wiele Powtórzeń i Analiza Statystyczna (Zalecane)

**⚠️ Ważne:** Jeden "run" = 30 powtórzeń tej samej konfiguracji eksperymentów

**Workflow:**

**1. Uruchom 30 powtórzeń** dla pierwszej konfiguracji:
```bash
for i in {1..30}; do 
    echo "Iteracja $i/30..."; 
    python main.py; 
done
```

**2. Uruchom analizę statystyczną** (pojedynczy run):
```bash
python ga_tsp_analyses.py
```
Wygeneruje:
- Statystyki stabilności eksperymentów (CV, mean, std)
- Wykresy porównawcze (boxplot, scatter plot, histogramy)
- Analizę w folderze `outputs/analysis/`

**3. Archiwizuj wyniki** (przed zmianą `config.yaml`!):
```bash
mv outputs outputs_run_1
mv experiment_results.csv experiments/experiment_results_run_1.csv
```

**4. Zmień konfigurację** w `config.yaml` (nowe eksperymenty)

**5. Powtórz kroki 1-3** dla `run_2`, `run_3`, itd.

**6. Uruchom analizę porównawczą** wszystkich runs:
```bash
python run_analyses.py
```

**📖 Szczegółowy przewodnik:**  
Zobacz [MULTI_RUN_GUIDE](MULTI_RUN_GUIDE) aby dowiedzieć się:
- Jak uruchomić wiele eksperymentów automatycznie
- Jak wyłączyć wizualizacje dla szybszego działania
- Jak organizować wyniki po każdym batch'u
- Przykładowe komendy dla Linux/Mac/Windows
- Jak unikać typowych błędów

**Wyniki znajdziesz w:** `analysis_results/`

---

### 📊 Wygenerowane Wykresy (Analiza Porównawcza)

`run_analyses.py` generuje 7 wykresów:
1. TOP 10 najlepszych pojedynczych wyników
2. Scatter plot: Stabilność vs Jakość
3. Trzy perspektywy TOP 5 (jakość, stabilność, kompromis)
4. Rozkład współczynnika zmienności (CV)
5. Wpływ parametrów (crossover, selection) na wyniki
6. Boxplot porównujący wszystkie eksperymenty
7. Heatmapa konsystencji eksperymentów

---

### Struktura Wyników - Pojedyncze Uruchomienie
```
outputs/
├── convergence/        # Wykresy zbieżności algorytmu
├── routes/            # Wizualizacje najlepszych tras
├── animations/        # Animacje GIF ewolucji rozwiązań
└── analysis/          # Statystyki i analizy porównawcze (po 30 iteracjach)
```

### Struktura Wyników - Analiza Wielu Runs
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
├── main.py                              # Run experiments
├── utils.py                             # Helper functions (TSP, GA, visualizations)
├── config.yaml                          # Experiment configuration
├── ga_tsp_analyses.py                   # Single CSV analysis (30 repetitions)
├── run_analyses.py                      # ⭐ Multi-run comparative analysis
├── utils_unified_analyses.py            # ⭐ Multi-run analysis functions
├── held_karp.py                         # Held-Karp algorithm (optimal route)
├── MULTI_RUN_GUIDE.md                   # 📖 Detailed guide
├── experiments/                         # Data from multiple runs
│   ├── experiment_results_run_1.csv
│   ├── experiment_results_run_2.csv
│   └── ...
├── outputs/                             # Current run results
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

### PART 1: Basic Execution

**Run experiments:**
```bash
python main.py
```
The script will automatically:
- Execute all experiments defined in `config.yaml`
- Generate visualizations in the `outputs/` folder
- Append results to `experiment_results.csv`

**💡 Note:** Single execution of `main.py` generates charts, but **statistical analysis makes NO SENSE with just one result**. Multiple repetitions are needed (see PART 2).

---

### PART 2: Multiple Repetitions and Statistical Analysis (Recommended)

**⚠️ Important:** One "run" = 30 repetitions of the same experiment configuration

**Workflow:**

**1. Run 30 repetitions** for first configuration:
```bash
for i in {1..30}; do 
    echo "Iteration $i/30..."; 
    python main.py; 
done
```

**2. Run statistical analysis** (single run):
```bash
python ga_tsp_analyses.py
```
Generates:
- Experiment stability statistics (CV, mean, std)
- Comparative charts (boxplot, scatter plot, histograms)
- Analysis in `outputs/analysis/` folder

**3. Archive results** (before changing `config.yaml`!):
```bash
mv outputs outputs_run_1
mv experiment_results.csv experiments/experiment_results_run_1.csv
```

**4. Change configuration** in `config.yaml` (new experiments)

**5. Repeat steps 1-3** for `run_2`, `run_3`, etc.

**6. Run comparative analysis** of all runs:
```bash
python run_analyses.py
```

**📖 Detailed Guide:**  
See [MULTI_RUN_GUIDE.md](MULTI_RUN_GUIDE.md) to learn:
- How to run multiple experiments automatically
- How to disable visualizations for faster execution
- How to organize results after each batch
- Example commands for Linux/Mac/Windows
- How to avoid common mistakes

**Results will be in:** `analysis_results/`

---

### 📊 Generated Charts (Comparative Analysis)

`run_analyses.py` generates 7 charts:
1. TOP 10 best individual results
2. Scatter plot: Stability vs Quality
3. Three perspectives TOP 5 (quality, stability, compromise)
4. Coefficient of variation (CV) distribution
5. Parameter impact (crossover, selection) on results
6. Boxplot comparing all experiments
7. Experiment consistency heatmap

---

### Output Structure - Single Run
```
outputs/
├── convergence/        # Algorithm convergence plots
├── routes/            # Best route visualizations
├── animations/        # GIF animations of solution evolution
└── analysis/          # Statistics and comparative analysis (after 30 iterations)
```

### Output Structure - Multiple Runs Analysis
```
analysis_results/
├── data/              # 3 CSV files with aggregated data
├── charts/            # 7 PNG charts for presentations
└── summary_report.txt # Text report with conclusions
```

---

## 📊 Quick Tips

- Modify `config.yaml` to adjust experiment parameters
- One run = 30 repetitions of the SAME configuration (do not mix!)
- Disable visualizations in `config.yaml` when running 30 repetitions
- Always archive results BEFORE changing configuration
- `ga_tsp_analyses.py` requires at least 30 repetitions to be meaningful
- See [MULTI_RUN_GUIDE.md](MULTI_RUN_GUIDE.md) for detailed instructions
- All visualizations are saved at 300 DPI for presentation quality
