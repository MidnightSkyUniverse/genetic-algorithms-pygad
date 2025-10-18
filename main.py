"""
main.py - Główny skrypt do uruchamiania eksperymentów TSP
"""

import warnings
warnings.filterwarnings("ignore")

from utils import (
    load_config, 
    create_output_dirs, 
    merge_params,
    TSPProblem, 
    GAMonitor, 
    run_tsp_ga,
    plot_convergence,
    create_animation,
    plot_route_with_all_connections
)


def run_single_experiment(experiment, config, dirs, tsp_problem, monitor):
    """
    Uruchom pojedynczy eksperyment
    
    Args:
        experiment: Słownik z konfiguracją eksperymentu
        config: Pełna konfiguracja
        dirs: Słownik ze ścieżkami do folderów output
        tsp_problem: Instancja TSPProblem (wspólna dla wszystkich)
        monitor: Instancja GAMonitor
    """
    exp_name = experiment['name']
    exp_desc = experiment.get('description', '')
    
    print(f"\n{'='*70}")
    print(f"EKSPERYMENT: {exp_name}")
    print(f"Opis: {exp_desc}")
    print(f"{'='*70}")
    
    # Połącz parametry baseline z parametrami eksperymentu
    params = merge_params(config['baseline'], experiment['params'])


    # Pobierz najlepsze rozwiązanie
    ga_instance, best_solution, best_distance = run_tsp_ga(
        tsp_problem, monitor, params
    )

    # Wizualizacje
    viz_config = config['visualization']

    if viz_config['save_convergence']:
        convergence_path = dirs['convergence'] / f"{exp_name}_convergence.png"
        plot_convergence(
            monitor,
            ga_instance,
            title=f"Zbieżność: {exp_desc}",
            save_path=str(convergence_path)
        )

    if viz_config['save_animation']:
        animation_path = dirs['animations'] / f"{exp_name}_evolution.gif"
        create_animation(
            tsp_problem,
            monitor,
            interval_generations=viz_config['animation_interval'],
            save_path=str(animation_path)
        )

    if viz_config['save_route']:
        route_path = dirs['routes'] / f"{exp_name}_best_route.png"
        plot_route_with_all_connections(
            tsp_problem,
            best_solution,
            title="TSP Best Route Using GA",
            save_path=str(route_path)
        )



    return {
        'name': exp_name,
        'description': exp_desc,
        'params': params,
        'best_distance': best_distance,
        'generations': ga_instance.generations_completed
    }



def main():
    """Główna funkcja programu"""
    
    # Załaduj konfigurację
    config = load_config("config.yaml")
    
    # Stwórz foldery na outputy
    output_config = config['output']
    dirs = create_output_dirs(output_config['base_dir'])
    
    print("="*70)
    print("EKSPERYMENTY TSP Z ALGORYTMEM GENETYCZNYM")
    print("="*70)
    
    # Stwórz problem TSP (ten sam dla wszystkich eksperymentów)
    problem_config = config['problem']
    
    if problem_config['use_fixed_cities']:
        fixed_cities = config['fixed_cities']
        tsp_problem = TSPProblem(
            num_cities=problem_config['num_cities'],
            fixed_cities=fixed_cities
        )
        print(f"Używam ustalonych {problem_config['num_cities']} miast")
    else:
        tsp_problem = TSPProblem(num_cities=problem_config['num_cities'])
        print(f"Generuję losowe {problem_config['num_cities']} miast")
    
    # Monitor do zbierania danych
    monitor = GAMonitor()
    
    # Uruchom wszystkie eksperymenty
    results = []
    experiments = config['experiments']
    
    for i, experiment in enumerate(experiments, 1):
        print(f"\n\nEksperyment {i}/{len(experiments)}")
        
        result = run_single_experiment(
            experiment, 
            config, 
            dirs, 
            tsp_problem, 
            monitor
        )
        results.append(result)
    
    # Podsumowanie
    print("\n\n" + "="*70)
    print("PODSUMOWANIE WSZYSTKICH EKSPERYMENTÓW")
    print("="*70)
    
    print(f"\n{'Nazwa':<20} {'Opis':<30} {'Najlepsza trasa':>15} {'Generacje':>12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['name']:<20} "
              f"{result['description']:<30} "
              f"{result['best_distance']:>15.2f} "
              f"{result['generations']:>12}")
    
    # Znajdź najlepszy wynik
    best_result = min(results, key=lambda x: x['best_distance'])
    print(f"\n{'='*70}")
    print(f"NAJLEPSZY WYNIK: {best_result['name']}")
    print(f"Trasa: {best_result['best_distance']:.2f}")
    print(f"Generacji: {best_result['generations']}")
    print(f"{'='*70}\n")
    
    print(f"\nWszystkie wyniki zapisane w folderze: {output_config['base_dir']}/")
    print("Wizualizacje gotowe do użycia w webinarze!\n")

    # Dodaj run_id do każdego wyniku
    import random
    run_id = random.randint(10000, 99999)  # Unikalny ID dla tego uruchomienia
    for result in results:
        result['run_id'] = run_id

    # Zapisz wyniki do CSV
    from utils import save_results_to_csv
    save_results_to_csv(results, filename="experiment_results.csv")

    print(f"\nWszystkie wyniki zapisane w folderze: {output_config['base_dir']}/")
    print("Wizualizacje gotowe do użycia w webinarze!\n")


if __name__ == "__main__":
    main()
