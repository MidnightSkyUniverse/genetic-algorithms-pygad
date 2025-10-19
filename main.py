"""
main.py - Main script for running TSP experiments with Genetic Algorithms

This script orchestrates multiple GA experiments on the same TSP instance,
allowing comparison of different parameter configurations. Results are
visualized and saved for analysis.
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
    Execute a single GA experiment with specified parameters.

    This function runs one complete genetic algorithm experiment, generates
    visualizations, and returns the results for comparison.

    Args:
        experiment: Dictionary containing experiment configuration
        config: Full configuration dictionary
        dirs: Dictionary with paths to output folders
        tsp_problem: TSPProblem instance (shared across all experiments)
        monitor: GAMonitor instance for data collection

    Returns:
        Dictionary containing experiment results (name, parameters, best distance, etc.)
    """
    exp_name = experiment['name']
    exp_desc = experiment.get('description', '')

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"Description: {exp_desc}")
    print(f"{'='*70}")

    # Merge baseline parameters with experiment-specific parameters
    params = merge_params(config['baseline'], experiment['params'])

    # Run genetic algorithm
    ga_instance, best_solution, best_distance = run_tsp_ga(
        tsp_problem, monitor, params
    )

    # Generate visualizations based on configuration
    #
    # ⚠️  IMPORTANT FOR RUNNING 30 REPETITIONS:
    #
    # When running multiple iterations (e.g., for i in {1..30}; do python main.py; done):
    # - Visualizations are generated EVERY iteration for EVERY experiment
    # - Only the LAST iteration's charts remain (files are overwritten)
    # - Animations significantly slow down execution (5-10 seconds each)
    # - Most charts are unnecessary for statistical analysis
    #
    # RECOMMENDATION: Disable visualizations for 30-repetition runs:
    #
    # Option 1 - Edit config.yaml:
    #   visualization:
    #     save_convergence: false
    #     save_route: false
    #     save_animation: false      # ← This is the slowest part!
    #
    # Option 2 - Comment out the blocks below
    #
    # ═══════════════════════════════════════════════════════════════
    # WORKFLOW REMINDER:
    # ═══════════════════════════════════════════════════════════════
    # 1. Configure experiments in config.yaml (e.g., baseline, k3, k5, k7)
    # 2. Run 30 iterations: for i in {1..30}; do python main.py; done
    # 3. Archive results BEFORE changing config:
    #    mv outputs outputs_run_1
    #    mv experiment_results.csv experiments/experiment_results_run_1.csv
    # 4. Change config.yaml to NEW experiments (e.g., roulette, sus)
    # 5. Repeat steps 2-3 for run_2, run_3, etc.
    # 6. Run comparative analysis: python run_analyses.py
    #
    # Each "run" = 30 repetitions of the SAME configuration
    # DO NOT mix different configurations in one CSV!
    # ═══════════════════════════════════════════════════════════════

    viz_config = config['visualization']

    if viz_config['save_convergence']:
        convergence_path = dirs['convergence'] / f"{exp_name}_convergence.png"
        plot_convergence(
            monitor,
            ga_instance,
            title=f"Convergence: {exp_desc}",
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

    # Return experiment results
    return {
        'name': exp_name,
        'description': exp_desc,
        'params': params,
        'best_distance': best_distance,
        'generations': ga_instance.generations_completed
    }


def main():
    """
    Main program function - orchestrates all experiments.

    This function:
    1. Loads configuration from config.yaml
    2. Creates a single TSP problem instance (ensures fair comparison)
    3. Runs all configured experiments
    4. Saves results and generates summary
    """

    # Load configuration from YAML file
    config = load_config("config.yaml")

    # Create output directory structure
    output_config = config['output']
    dirs = create_output_dirs(output_config['base_dir'])

    print("="*70)
    print("TSP EXPERIMENTS WITH GENETIC ALGORITHM")
    print("="*70)

    # Create TSP problem instance (same for all experiments to ensure fair comparison)
    problem_config = config['problem']

    if problem_config['use_fixed_cities']:
        fixed_cities = config['fixed_cities']
        tsp_problem = TSPProblem(
            num_cities=problem_config['num_cities'],
            fixed_cities=fixed_cities
        )
        print(f"Using fixed set of {problem_config['num_cities']} cities")
    else:
        tsp_problem = TSPProblem(num_cities=problem_config['num_cities'])
        print(f"Generating random {problem_config['num_cities']} cities")

    # Create monitor for collecting data across experiments
    monitor = GAMonitor()

    # Run all configured experiments
    results = []
    experiments = config['experiments']

    for i, experiment in enumerate(experiments, 1):
        print(f"\n\nExperiment {i}/{len(experiments)}")

        result = run_single_experiment(
            experiment,
            config,
            dirs,
            tsp_problem,
            monitor
        )
        results.append(result)

    # Print summary of all experiments
    print("\n\n" + "="*70)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("="*70)

    print(f"\n{'Name':<20} {'Description':<30} {'Best Distance':>15} {'Generations':>12}")
    print("-" * 80)

    for result in results:
        print(f"{result['name']:<20} "
              f"{result['description']:<30} "
              f"{result['best_distance']:>15.2f} "
              f"{result['generations']:>12}")

    # Identify best performing configuration
    best_result = min(results, key=lambda x: x['best_distance'])
    print(f"\n{'='*70}")
    print(f"BEST RESULT: {best_result['name']}")
    print(f"Distance: {best_result['best_distance']:.2f}")
    print(f"Generations: {best_result['generations']}")
    print(f"{'='*70}\n")

    # Assign unique run ID to this batch of experiments
    import random
    run_id = random.randint(10000, 99999)  # Unique ID for tracking multiple runs
    for result in results:
        result['run_id'] = run_id

    # Save results to CSV (appends to existing file for accumulation)
    from utils import save_results_to_csv
    save_results_to_csv(results, filename="experiment_results.csv")

    print(f"\nAll results saved to: {output_config['base_dir']}/")
    print("Visualizations ready for use in webinar!\n")


if __name__ == "__main__":
    main()