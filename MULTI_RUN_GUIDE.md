# Running Multiple Experiment Runs

## 🔍 Understanding the Workflow

**Key Concept:**
- **One "run"** = 30 repetitions of the SAME set of experiments (defined in `config.yaml`)
- **Multiple "runs"** = Testing DIFFERENT experiment configurations
- Each `experiment_results_run_X.csv` contains data from ONE configuration set only

**Example:**
- `run_1`: 30 repetitions of [baseline, k3, k7, roulette, sus] → `experiment_results_run_1.csv`
- `run_2`: 30 repetitions of [baseline, k7, population size ] → `experiment_results_run_2.csv`
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
