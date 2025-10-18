



projekt/
├── experiments/                          # Folder z danymi
│   ├── experiment_results_run_1.csv
│   ├── experiment_results_run_2.csv
│   └── ...
├── utils_unified_analyses.py             # ⭐ Wszystkie funkcje do finalnej analizy
├── run_analysis.py                       # ⭐ Krótki skrypt uruchamiający
└── analysis_results/                     # Folder wynikowy (utworzy się automatycznie)
    ├── data/
    │   ├── best_runs_per_experiment.csv
    │   ├── stability_per_run.csv
    │   └── full_experiment_data.csv
    ├── charts/
    │   ├── chart1_top10_best_results.png
    │   ├── chart2_stability_vs_quality.png
    │   ├── ... (5 więcej)
    └── summary_report.txt