# Run multiple experiments

## 🇵🇱 Polish Version

## 🔍 Zrozumienie Workflow

**Kluczowa koncepcja:**
- **Jeden "run"** = 30 powtórzeń TYCH SAMYCH eksperymentów (zdefiniowanych w `config.yaml`)
- **Wiele "runs"** = Testowanie RÓŻNYCH konfiguracji eksperymentów
- Każdy `experiment_results_run_X.csv` zawiera dane tylko z JEDNEGO zestawu konfiguracji

**Przykład:**
- `run_1`: 30 powtórzeń [baseline, k3, k5, k7] → `experiment_results_run_1.csv`
- `run_2`: 30 powtórzeń [baseline_v2, k9, roulette, sus] → `experiment_results_run_2.csv`
- NIE MIESZAJ różnych konfiguracji w tym samym pliku CSV!

---

## ⚠️ Ważne Uwagi o Wizualizacjach

**Skrypt `main.py` generuje wizualizacje (wykresy zbieżności, mapy tras, animacje) dla KAŻDEGO eksperymentu w KAŻDEJ iteracji.**

**Problem:** Przy 30 iteracjach:
- Wizualizacje są generowane 30 razy
- Zostają tylko wykresy z OSTATNIEJ iteracji (pliki są nadpisywane)
- Animacje znacząco spowalniają wykonanie
- Większość wykresów jest niepotrzebna do analizy statystycznej

### Opcje Wyłączenia Wizualizacji:

**Opcja 1: Modyfikacja `config.yaml`** (Zalecane)
```yaml
visualization:
  save_convergence: false    # Wyłącz wykresy zbieżności
  save_route: false          # Wyłącz wizualizacje tras
  save_animation: false      # Wyłącz animacje GIF (najwolniejsze!)
  animation_interval: 5
  dpi: 300
```

**Opcja 2: Zakomentowanie kodu wizualizacji w `main.py`**

Znajdź sekcję wizualizacji w funkcji `run_single_experiment()`:
```python
# Visualizations
viz_config = config['visualization']

# if viz_config['save_convergence']:
#     convergence_path = dirs['convergence'] / f"{exp_name}_convergence.png"
#     plot_convergence(...)

# if viz_config['save_animation']:
#     animation_path = dirs['animations'] / f"{exp_name}_evolution.gif"
#     create_animation(...)

# if viz_config['save_route']:
#     route_path = dirs['routes'] / f"{exp_name}_best_route.png"
#     plot_route_with_all_connections(...)
```

---

## 📋 Kompletny Workflow Krok po Kroku

### Krok 1: Skonfiguruj Pierwszą Serię Eksperymentów

Edytuj `config.yaml` aby zdefiniować eksperymenty:
```yaml
experiments:
  - name: "baseline"
    description: "Baseline Tournament (K=3)"
    params: {}
  
  - name: "k5"
    description: "Tournament K=5"
    params:
      K_tournament: 5
  
  - name: "k7"
    description: "Tournament K=7"
    params:
      K_tournament: 7
```

### Krok 2: Uruchom 30 Powtórzeń

**Linux/Mac:**
```bash
for i in {1..30}; do 
    echo "Uruchamianie iteracji $i/30..."; 
    python main.py; 
done
```

**Windows (PowerShell):**
```powershell
for ($i=1; $i -le 30; $i++) {
    Write-Host "Uruchamianie iteracji $i/30..."
    python main.py
}
```

**Windows (Wiersz polecenia):**
```batch
for /L %i in (1,1,30) do (
    echo Uruchamianie iteracji %i/30...
    python main.py
)
```

**Co się dzieje:**
- Każda iteracja testuje WSZYSTKIE eksperymenty z `config.yaml`
- Wyniki są dodawane do `experiment_results.csv`
- Wizualizacje w `outputs/` są nadpisywane za każdym razem
- Po 30 iteracjach: CSV zawiera 30 × (liczba eksperymentów) wierszy

### Krok 3: Archiwizuj Wyniki dla Tej Konfiguracji

**KRYTYCZNE:** Przenieś wyniki przed zmianą config.yaml!

```bash
# Zmień nazwę folderu outputs
mv outputs outputs_run_1

# Przenieś CSV do folderu experiments
mkdir -p experiments
mv experiment_results.csv experiments/experiment_results_run_1.csv
```

**Twoja struktura folderów teraz:**
```
projekt/
├── outputs_run_1/                        # Zarchiwizowane wykresy (ostatnia iteracja)
│   ├── convergence/
│   ├── routes/
│   └── animations/
└── experiments/
    └── experiment_results_run_1.csv      # Wszystkie 30 powtórzeń
```

### Krok 4: Zmień Konfigurację dla Następnego Run'a

Edytuj `config.yaml` z NOWYMI eksperymentami:
```yaml
experiments:
  - name: "roulette"
    description: "Roulette Wheel Selection"
    params:
      parent_selection_type: "rws"
  
  - name: "sus"
    description: "Stochastic Universal Sampling"
    params:
      parent_selection_type: "sss"
```

### Krok 5: Powtórz Kroki 2-3 dla Każdej Nowej Konfiguracji

```bash
# Uruchom serię 2 (30 iteracji NOWYCH eksperymentów)
for i in {1..30}; do echo "Uruchamianie iteracji $i/30..."; python main.py; done
mv outputs outputs_run_2
mv experiment_results.csv experiments/experiment_results_run_2.csv

# Uruchom serię 3 (zmień config.yaml ponownie, potem uruchom)
for i in {1..30}; do echo "Uruchamianie iteracji $i/30..."; python main.py; done
mv outputs outputs_run_3
mv experiment_results.csv experiments/experiment_results_run_3.csv

# ... i tak dalej
```

### Krok 6: Uruchom Analizę Porównawczą Wielu Runs

Gdy masz wiele plików `experiment_results_run_*.csv`:

```bash
python run_analyses.py
```

To:
- Załaduje WSZYSTKIE pliki CSV z folderu `experiments/`
- Porówna RÓŻNE konfiguracje eksperymentów między runs
- Wygeneruje 7 kompleksowych wykresów
- Stworzy raport statystyczny

---

## 📂 Oczekiwana Struktura Folderów

Po ukończeniu 3 różnych serii konfiguracji:

```
projekt/
├── main.py
├── run_analyses.py
├── config.yaml
├── experiments/                          # ← Pliki CSV (jeden na serię konfiguracji)
│   ├── experiment_results_run_1.csv     # Run 1: baseline, k3, k5, k7 (30 powtórzeń każdy)
│   ├── experiment_results_run_2.csv     # Run 2: roulette, sus, rank (30 powtórzeń każdy)
│   └── experiment_results_run_3.csv     # Run 3: testy ox_crossover (30 powtórzeń każdy)
├── outputs_run_1/                        # ← Zarchiwizowane wizualizacje
│   ├── convergence/
│   ├── routes/
│   └── animations/
├── outputs_run_2/
│   ├── convergence/
│   ├── routes/
│   └── animations/
├── outputs_run_3/
│   ├── convergence/
│   ├── routes/
│   └── animations/
└── analysis_results/                     # ← Wygenerowane przez run_analyses.py
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

## 💡 Wskazówki dla Efektywnych Eksperymentów

1. **Zawsze wyłączaj wizualizacje** przy uruchamianiu 30 powtórzeń
   - Oszczędza czas: ~5-10 sekund na eksperyment na iterację
   - Dla 4 eksperymentów × 30 iteracji: oszczędza ~10-20 minut
   - Animacje są najwolniejsze (5-10 MB każda, trwa sekundy)

2. **Archiwizuj wyniki NATYCHMIAST** po ukończeniu serii
   - Nie zapomnij przenieść zarówno `outputs/` jak i `experiment_results.csv`
   - Mieszanie konfiguracji w jednym CSV psuje analizę

3. **Używaj opisowych numerów run** w swoich notatkach
   - Dokumentuj co testuje każdy run (np. run_1 = porównanie tournament)
   - Pomaga przy późniejszej analizie wyników

4. **Zacznij od mniejszej liczby powtórzeń** do testowania
   - Spróbuj 5-10 powtórzeń najpierw aby zweryfikować konfigurację
   - Potem uruchom pełne 30 powtórzeń na noc

5. **Zachowaj pliki CSV, archiwizuj lub usuń foldery outputs**
   - CSV zawiera wszystkie istotne dane
   - Foldery `outputs_run_X/` są duże ale można je regenerować jeśli potrzeba
   - Można usunąć stare foldery outputs aby zaoszczędzić miejsce na dysku

6. **Uruchamiaj eksperymenty na noc lub podczas przerw**
   - 30 powtórzeń × 4 eksperymenty × 2-3 minuty ≈ 2-4 godziny

---

## 🎯 Diagram Workflow

```
┌──────────────────────────────────────────┐
│ 1. Edytuj config.yaml                    │
│    Zdefiniuj zestaw eksperymentów        │
│    (np. k3-k7)                           │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│ 2. Wyłącz wizualizacje (opcjonalnie)     │
│    Ustaw save_animation: false w config  │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│ 3. Uruchom 30 iteracji                   │
│    for i in {1..30}; do python main.py   │
│    → Tworzy experiment_results.csv       │
│    → Tworzy outputs/ (nadpisywane)       │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│ 4. Archiwizuj wyniki NATYCHMIAST         │
│    mv outputs → outputs_run_1            │
│    mv CSV → experiments/...run_1.csv     │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│ 5. Zmień config.yaml na następną serię   │
│    Zdefiniuj NOWE eksperymenty           │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│ 6. Powtórz kroki 3-5 dla każdej config   │
│    run_2, run_3, run_4, itd.             │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│ 7. Uruchom analizę porównawczą           │
│    python run_analyses.py                │
│    → Porównuje WSZYSTKIE runs            │
└──────────────────────────────────────────┘
```

---

## ❓ Częste Problemy

**Problem**: "experiment_results.csv już istnieje, dane są dodawane"
- **Rozwiązanie**: To jest NORMALNE podczas 30 iteracji
- Przenieś CSV dopiero po ukończeniu WSZYSTKICH 30 iteracji
- Nigdy nie mieszaj różnych konfiguracji w jednym CSV!

**Problem**: Iteracje są bardzo wolne
- **Rozwiązanie**: Wyłącz animacje w `config.yaml` (największa oszczędność czasu)
- Każda animacja zajmuje 5-10 sekund i 5-10 MB
- Wykresy zbieżności i wizualizacje tras też dodają czasu

**Problem**: Zapomniałem przenieść plików, uruchomiłem nową konfigurację
- **Rozwiązanie**: CSV ma teraz ZMIESZANE dane - musisz go usunąć i zacząć od nowa
- Zawsze archiwizuj PRZED zmianą config.yaml!

**Problem**: `run_analyses.py` pokazuje dziwne wyniki
- **Rozwiązanie**: Sprawdź czy każdy plik CSV zawiera tylko JEDEN zestaw konfiguracji
- Zweryfikuj nazewnictwo plików: `experiment_results_run_1.csv`, nie `run1.csv`
- Każdy run powinien mieć spójne nazwy eksperymentów przez wszystkie 30 powtórzeń

**Problem**: Brak miejsca na dysku
- **Rozwiązanie**: Usuń stare foldery `outputs_run_X/` (szczególnie animations/)
- Zachowaj tylko pliki CSV - zawierają wszystkie istotne dane
- Pliki CSV są małe (~1-5 MB), animacje są duże (~50-200 MB na run)

---

## 📊 Co Powinien Zawierać Każdy Run

Każdy plik `experiment_results_run_X.csv` powinien mieć:
- **Te same nazwy eksperymentów** powtórzone 30 razy
- **Te same parametry** dla każdego eksperymentu (oprócz random seed)
- **Różne wartości best_distance** (ze względu na stochastyczność algorytmu)

Przykładowa struktura:
```
timestamp,run_id,experiment_name,best_distance,...
2025-01-15,12345,baseline,425.67,...
2025-01-15,12345,k5,418.23,...
2025-01-15,12345,k7,412.89,...
2025-01-15,12345,baseline,428.91,...  ← iteracja 2
2025-01-15,12345,k5,415.67,...        ← iteracja 2
...
(30 powtórzeń dla każdego eksperymentu)
```

**NIE MIESZAJ** konfiguracji typu:
```
❌ ŹLE:
baseline (K=3), k5, k7,          ← z run 1
roulette, sus,                   ← z run 2 (zmieszane!)
ox_crossover                     ← z run 3 (zmieszane!)
```

---

## 🔬 Dodatkowe Wskazówki

### Optymalizacja Czasu
- Wyłącz `save_animation: false` - największa oszczędność
- Wyłącz `save_convergence: false` - średnia oszczędność
- Pozostaw `save_route: true` - przydatne, nie spowalnia znacząco

### Organizacja Danych
- Twórz backup config.yaml dla każdego run'a
- Nazywaj: `config_run_1.yaml`, `config_run_2.yaml`
- Pomaga zapamiętać co testował każdy run

### Dokumentacja
- Prowadź notes z datami i celami każdego run'a
- Zapisz hipotezy przed uruchomieniem eksperymentów
- Notuj obserwacje po analizie wyników

### Testowanie
- **5 powtórzeń** - szybki test czy konfiguracja działa
- **10 powtórzeń** - wstępna analiza stabilności
- **30 powtórzeń** - pełna analiza statystyczna
- **50+ powtórzeń** - gdy potrzebujesz bardzo wysokiej precyzji

---

## 📖 Kolejne Kroki

Po zebraniu danych z wielu runs:

1. **Uruchom `run_analyses.py`** - generuje kompleksowe porównanie
2. **Przeanalizuj wykresy** w `analysis_results/charts/`
3. **Przeczytaj raport** w `analysis_results/summary_report.txt`
4. **Zidentyfikuj najlepszą konfigurację** dla swojego problemu
5. **Stwórz prezentację** używając wygenerowanych wykresów

**Pamiętaj:** Celem nie jest znalezienie jednego "najlepszego" wyniku, ale **zrozumienie** które parametry wpływają na jakość i stabilność algorytmu genetycznego dla problemu TSP.

---

## 🎓 Dla Studentów

### Kluczowe Zasady Eksperymentowania

1. **Jedna zmiana na raz**
   - Zmieniaj tylko jeden parametr między runs
   - Tak izolujesz wpływ konkretnego parametru

2. **Dokumentuj wszystko**
   - Co zmieniłeś
   - Dlaczego to zmieniłeś
   - Czego się spodziewasz

3. **Analizuj przed kolejnym krokiem**
   - Nie uruchamiaj wszystkich runs na ślepo
   - Analizuj wyniki, formułuj hipotezy
   - Projektuj następny run na podstawie wniosków

4. **Statystyka > Pojedyncze wyniki**
   - 30 powtórzeń daje ci statystyczną pewność
   - Jeden dobry wynik może być szczęściem
   - Niski CV = powtarzalne rezultaty

5. **Czas to zasób**
   - Zaplanuj eksperymenty przed uruchomieniem
   - Wyłącz niepotrzebne wizualizacje
   - Uruchamiaj na noc gdy to możliwe

Powodzenia w eksperymentach! 🚀



# Running Multiple Experiment Runs

## 🇬🇧 English Version

## 🔍 Understanding the Workflow

**Key Concept:**
- **One "run"** = 30 repetitions of the SAME set of experiments (defined in `config.yaml`)
- **Multiple "runs"** = Testing DIFFERENT experiment configurations
- Each `experiment_results_run_X.csv` contains data from ONE configuration set only

**Example:**
- `run_1`: 30 repetitions of [baseline, k3, k5, k7] → `experiment_results_run_1.csv`
- `run_2`: 30 repetitions of [baseline_v2, k9, roulette, sus] → `experiment_results_run_2.csv`
- DO NOT mix different configurations in the same CSV file!

---

## ⚠️ Important Notes About Visualizations

**The `main.py` script generates visualizations (convergence plots, route maps, animations) for EACH experiment in EACH iteration.**

**Problem:** When running 30 iterations:
- Visualizations are generated 30 times
- Only the LAST iteration's charts remain (files are overwritten)
- Animations significantly slow down execution
- Most charts are unnecessary for statistical analysis

### Options to Disable Visualizations:

**Option 1: Modify `config.yaml`** (Recommended)
```yaml
visualization:
  save_convergence: false    # Disable convergence plots
  save_route: false          # Disable route visualizations
  save_animation: false      # Disable GIF animations (slowest part!)
  animation_interval: 5
  dpi: 300
```

**Option 2: Comment out visualization code in `main.py`**

Locate the visualization section in `run_single_experiment()` function:
```python
# Visualizations
viz_config = config['visualization']

# if viz_config['save_convergence']:
#     convergence_path = dirs['convergence'] / f"{exp_name}_convergence.png"
#     plot_convergence(...)

# if viz_config['save_animation']:
#     animation_path = dirs['animations'] / f"{exp_name}_evolution.gif"
#     create_animation(...)

# if viz_config['save_route']:
#     route_path = dirs['routes'] / f"{exp_name}_best_route.png"
#     plot_route_with_all_connections(...)
```

---

## 📋 Complete Step-by-Step Workflow

### Step 1: Configure First Batch of Experiments

Edit `config.yaml` to define your experiments:
```yaml
experiments:
  - name: "baseline"
    description: "Baseline Tournament (K=3)"
    params: {}

  - name: "k5"
    description: "Tournament K=5"
    params:
      K_tournament: 5

  - name: "k7"
    description: "Tournament K=7"
    params:
      K_tournament: 7
```

### Step 2: Run 30 Repetitions

**Linux/Mac:**
```bash
for i in {1..30}; do
    echo "Running iteration $i/30...";
    python main.py;
done
```

**Windows (PowerShell):**
```powershell
for ($i=1; $i -le 30; $i++) {
    Write-Host "Running iteration $i/30..."
    python main.py
}
```

**Windows (Command Prompt):**
```batch
for /L %i in (1,1,30) do (
    echo Running iteration %i/30...
    python main.py
)
```

**What happens:**
- Each iteration tests ALL experiments from `config.yaml`
- Results are appended to `experiment_results.csv`
- Visualizations in `outputs/` are overwritten each time
- After 30 iterations: CSV has 30 × (number of experiments) rows

### Step 3: Archive Results for This Configuration

**CRITICAL:** Move results before changing config.yaml!

```bash
# Rename outputs folder
mv outputs outputs_run_1

# Move CSV to experiments folder
mkdir -p experiments
mv experiment_results.csv experiments/experiment_results_run_1.csv
```

**Your folder structure now:**
```
project/
├── outputs_run_1/                        # Archived charts (last iteration)
│   ├── convergence/
│   ├── routes/
│   └── animations/
└── experiments/
    └── experiment_results_run_1.csv      # All 30 repetitions
```

### Step 4: Change Configuration for Next Run

Edit `config.yaml` with NEW experiments:
```yaml
experiments:
  - name: "roulette"
    description: "Roulette Wheel Selection"
    params:
      parent_selection_type: "rws"

  - name: "sus"
    description: "Stochastic Universal Sampling"
    params:
      parent_selection_type: "sss"
```

### Step 5: Repeat Steps 2-3 for Each New Configuration

```bash
# Run batch 2 (30 iterations of NEW experiments)
for i in {1..30}; do echo "Running iteration $i/30..."; python main.py; done
mv outputs outputs_run_2
mv experiment_results.csv experiments/experiment_results_run_2.csv

# Run batch 3 (change config.yaml again, then run)
for i in {1..30}; do echo "Running iteration $i/30..."; python main.py; done
mv outputs outputs_run_3
mv experiment_results.csv experiments/experiment_results_run_3.csv

# ... and so on
```

### Step 6: Run Multi-Run Comparative Analysis

Once you have multiple `experiment_results_run_*.csv` files:

```bash
python run_analyses.py
```

This will:
- Load ALL CSV files from `experiments/` folder
- Compare DIFFERENT experiment configurations across runs
- Generate 7 comprehensive charts
- Create statistical report

---

## 📂 Expected Folder Structure

After completing 3 different configuration batches:

```
project/
├── main.py
├── run_analyses.py
├── config.yaml
├── experiments/                          # ← CSV files (one per configuration batch)
│   ├── experiment_results_run_1.csv     # Run 1: baseline, k3, k5, k7 (30 reps each)
│   ├── experiment_results_run_2.csv     # Run 2: roulette, sus, rank (30 reps each)
│   └── experiment_results_run_3.csv     # Run 3: ox_crossover tests (30 reps each)
├── outputs_run_1/                        # ← Archived visualizations
│   ├── convergence/
│   ├── routes/
│   └── animations/
├── outputs_run_2/
│   ├── convergence/
│   ├── routes/
│   └── animations/
├── outputs_run_3/
│   ├── convergence/
│   ├── routes/
│   └── animations/
└── analysis_results/                     # ← Generated by run_analyses.py
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

## 💡 Tips for Efficient Multi-Run Experiments

1. **Always disable visualizations** when running 30 repetitions
   - Saves 5-10 seconds per experiment per iteration
   - For 4 experiments × 30 iterations: saves ~10-20 minutes
   - Animations are the slowest (5-10 MB each, takes seconds to generate)

2. **Archive results IMMEDIATELY** after completing a batch
   - Don't forget to `mv` both `outputs/` and `experiment_results.csv`
   - Mixing configurations in one CSV breaks the analysis

3. **Use descriptive run numbers** in your notes
   - Document what each run tests (e.g., run_1 = tournament comparison)
   - Helps when analyzing results later

4. **Start with fewer repetitions** for testing
   - Try 5-10 repetitions first to verify configuration
   - Then run full 30 repetitions overnight

5. **Keep CSV files, archive or delete outputs folders**
   - CSV contains all essential data
   - `outputs_run_X/` folders are large but can be regenerated if needed
   - Can delete old outputs folders to save disk space

6. **Run experiments overnight or during breaks**
   - 30 repetitions × 4 experiments × 2-3 minutes ≈ 2-4 hours

---

## 🎯 Workflow Diagram

```
┌──────────────────────────────────────────┐
│ 1. Edit config.yaml                      │
│    Define experiment set (e.g., k3-k7)   │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│ 2. Disable visualizations (optional)     │
│    Set save_animation: false in config   │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│ 3. Run 30 iterations                     │
│    for i in {1..30}; do python main.py   │
│    → Creates experiment_results.csv      │
│    → Creates outputs/ (overwritten)      │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│ 4. Archive results IMMEDIATELY           │
│    mv outputs → outputs_run_1            │
│    mv CSV → experiments/...run_1.csv     │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│ 5. Change config.yaml for next batch     │
│    Define NEW experiments                │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│ 6. Repeat steps 3-5 for each config      │
│    run_2, run_3, run_4, etc.             │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│ 7. Run multi-run comparative analysis    │
│    python run_analyses.py                │
│    → Compares ALL runs                   │
└──────────────────────────────────────────┘
```

---

## ❓ Common Issues

**Problem**: "experiment_results.csv already exists, data is being appended"
- **Solution**: This is NORMAL during the 30 iterations
- Only move the CSV after completing ALL 30 iterations
- Never mix different configurations in one CSV!

**Problem**: Iterations are very slow
- **Solution**: Disable animations in `config.yaml` (biggest time saver)
- Each animation takes 5-10 seconds and 5-10 MB
- Convergence plots and route visualizations also add time

**Problem**: Forgot to move files, ran new configuration
- **Solution**: The CSV now has MIXED data - you must delete it and start over
- Always archive BEFORE changing config.yaml!

**Problem**: `run_analyses.py` shows strange results
- **Solution**: Check that each CSV file contains only ONE configuration set
- Verify file naming: `experiment_results_run_1.csv`, not `run1.csv`
- Each run should have consistent experiment names across all 30 repetitions

**Problem**: Not enough disk space
- **Solution**: Delete old `outputs_run_X/` folders (especially animations/)
- Keep only the CSV files - they contain all essential data
- CSV files are small (~1-5 MB), animations are large (~50-200 MB per run)

---

## 📊 What Each Run Should Contain

Each `experiment_results_run_X.csv` file should have:
- **Same experiment names** repeated 30 times
- **Same parameters** for each experiment (except random seed)
- **Different best_distance values** (due to algorithm stochasticity)

Example structure:
```
timestamp,run_id,experiment_name,best_distance,...
2025-01-15,12345,baseline,425.67,...
2025-01-15,12345,k5,418.23,...
2025-01-15,12345,k7,412.89,...
2025-01-15,12345,baseline,428.91,...  ← iteration 2
2025-01-15,12345,k5,415.67,...        ← iteration 2
...
(30 repetitions for each experiment)
```

**DO NOT mix** configurations like:
```
❌ WRONG:
baseline (K=3), k5, k7,          ← from run 1
roulette, sus,                   ← from run 2 (mixed!)
ox_crossover                     ← from run 3 (mixed!)
```
