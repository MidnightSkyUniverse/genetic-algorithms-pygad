"""
utils.py - Helper functions and classes for TSP with Genetic Algorithm

This module contains:
- TSPProblem: Class representing the Traveling Salesman Problem instance
- GAMonitor: Class for collecting data during GA execution
- Helper functions for configuration, visualization, and result saving
- Custom crossover operators optimized for TSP
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import pygad
import yaml
from pathlib import Path
import csv
from datetime import datetime


# ============================================================================
# TSP PROBLEM CLASS
# ============================================================================

class TSPProblem:
    """
    Represents a Traveling Salesman Problem instance.

    This class manages city coordinates and calculates distances between cities.
    It can work with either fixed city locations or randomly generated ones.
    """

    def __init__(self, num_cities=15, fixed_cities=None, seed=42):
        """
        Initialize TSP problem instance.

        Args:
            num_cities: Number of cities in the problem
            fixed_cities: Numpy array with predefined city coordinates (optional)
            seed: Random seed for reproducibility when generating random cities
        """
        self.num_cities = num_cities

        if fixed_cities is not None:
            self.cities = np.array(fixed_cities)
        else:
            np.random.seed(seed)
            self.cities = np.random.rand(num_cities, 2) * 100

        self.distance_matrix = self._calculate_distance_matrix()

    def _calculate_distance_matrix(self):
        """
        Calculate Euclidean distance matrix between all city pairs.

        Returns:
            Symmetric matrix where element [i,j] is the distance between city i and j
        """
        n = self.num_cities
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(
                    (self.cities[i, 0] - self.cities[j, 0])**2 +
                    (self.cities[i, 1] - self.cities[j, 1])**2
                )
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        return dist_matrix

    def calculate_route_distance(self, route):
        """
        Calculate total distance of a given route.

        Args:
            route: Array of city indices representing the tour order

        Returns:
            Total distance of the complete tour (including return to start)
        """
        distance = 0
        for i in range(len(route) - 1):
            distance += self.distance_matrix[int(route[i]), int(route[i+1])]
        # Add distance back to starting city
        distance += self.distance_matrix[int(route[-1]), int(route[0])]
        return distance


# ============================================================================
# DATA MONITORING
# ============================================================================

class GAMonitor:
    """
    Collects and stores data during Genetic Algorithm execution.

    This class tracks the evolution of solutions across generations,
    storing best and average fitness values along with the actual routes.
    """

    def __init__(self):
        self.generation_numbers = []
        self.best_distances = []
        self.avg_distances = []
        self.best_routes = []

    def reset(self):
        """Clear all collected data (useful for running multiple experiments)."""
        self.generation_numbers = []
        self.best_distances = []
        self.avg_distances = []
        self.best_routes = []

    def record_generation(self, ga_instance, tsp_problem):
        """
        Record data from current generation.

        Args:
            ga_instance: PyGAD GA instance with current population
            tsp_problem: TSPProblem instance for calculating distances
        """
        # Calculate distances for all solutions in current population
        distances = []
        for solution in ga_instance.population:
            dist = tsp_problem.calculate_route_distance(solution)
            distances.append(dist)
            # Validate solution (check for duplicate cities - should not happen)
            if len(solution) != len(set(solution)):
                print(f"⚠️ Invalid solution in gen {ga_instance.generations_completed}: {solution}")

        # Find best solution in current generation
        best_idx = np.argmin(distances)
        best_distance = distances[best_idx]
        best_solution = ga_instance.population[best_idx].copy()

        # Calculate population statistics
        population_distances = [
            tsp_problem.calculate_route_distance(sol)
            for sol in ga_instance.population
        ]
        avg_distance = np.mean(population_distances)

        # Store data
        self.generation_numbers.append(ga_instance.generations_completed)
        self.best_distances.append(best_distance)
        self.avg_distances.append(avg_distance)
        self.best_routes.append(best_solution.copy())


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_config(config_path="config.yaml"):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing all configuration parameters
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_output_dirs(base_dir="outputs"):
    """
    Create directory structure for output files.

    Args:
        base_dir: Base directory name for all outputs

    Returns:
        Dictionary with paths to all output subdirectories
    """
    dirs = {
        'base': Path(base_dir),
        'convergence': Path(base_dir) / 'convergence',
        'routes': Path(base_dir) / 'routes',
        'animations': Path(base_dir) / 'animations'
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def merge_params(baseline, experiment_params):
    """
    Merge baseline parameters with experiment-specific parameters.

    Experiment parameters override baseline values where specified.

    Args:
        baseline: Dictionary with default/baseline parameters
        experiment_params: Dictionary with experiment-specific overrides

    Returns:
        Merged parameter dictionary
    """
    params = baseline.copy()
    params.update(experiment_params)
    return params


# ============================================================================
# GENETIC ALGORITHM EXECUTION
# ============================================================================

def run_tsp_ga(tsp_problem, monitor, params):
    """
    Execute Genetic Algorithm for TSP with given parameters.

    Args:
        tsp_problem: TSPProblem instance
        monitor: GAMonitor instance for data collection
        params: Dictionary containing GA parameters

    Returns:
        Tuple of (ga_instance, best_solution, best_distance)
    """
    monitor.reset()

    # Fitness function - higher is better (inverse of distance)
    def fitness_func(ga_instance, solution, solution_idx):
        distance = tsp_problem.calculate_route_distance(solution)
        return 1.0 / (distance + 1e-6)  # Add small epsilon to avoid division by zero

    # Callback executed after each generation
    def on_generation(ga_instance):
        monitor.record_generation(ga_instance, tsp_problem)

    # Select crossover operator
    if params['crossover_type'] == 'ox':
        crossover_func = crossover_func_ox
    else:
        crossover_func = params['crossover_type']

    # Configure PyGAD instance
    ga_instance = pygad.GA(
        num_generations=params['num_generations'],
        num_parents_mating=params['num_parents_mating'],
        sol_per_pop=params['population_size'],
        num_genes=tsp_problem.num_cities,

        fitness_func=fitness_func,

        # Gene configuration for permutation encoding
        gene_type=int,
        gene_space=list(range(tsp_problem.num_cities)),
        allow_duplicate_genes=False,  # Critical for TSP - each city visited once

        # Parent selection
        parent_selection_type=params['parent_selection_type'],
        K_tournament=params.get('K_tournament'),

        # Crossover
        crossover_type=crossover_func,
        crossover_probability=params.get('crossover_probability'),

        # Mutation
        mutation_type=params.get('mutation_type', 'swap'),
        mutation_percent_genes=params['mutation_percent_genes'],
        mutation_probability=params.get('mutation_probability'),

        # Elitism - preserve best solutions
        keep_elitism=params.get('keep_elitism'),

        on_generation=on_generation,

        # Stopping criteria
        stop_criteria=params.get('stop_criteria', 'saturate_50'),

        save_best_solutions=True,
    )

    # Run the genetic algorithm
    ga_instance.run()

    # Find overall best solution across all generations
    best_distance = float('inf')
    for idx, sol in enumerate(ga_instance.best_solutions):
        dist = tsp_problem.calculate_route_distance(sol)
        if dist < best_distance:
            best_distance = dist
            best_solution = sol.copy()
            best_generation = idx

    print(f"Best route distance: {best_distance:.2f}")
    print(f"Completed {ga_instance.generations_completed} generations")

    return ga_instance, best_solution, best_distance


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

# Color palette for consistent visual style
COLORS = {
    'primary': '#5c26ff',      # Purple primary
    'secondary': '#9c31ff',    # Purple secondary
    'accent': '#ffba0a',       # Yellow accent
    'start': '#00ff88',        # Green for start (optional)
    'connections': '#E0E0E0',  # Light gray for all connections
}


def plot_convergence(monitor, ga_instance, title="GA Convergence", save_path=None):
    """
    Visualize algorithm convergence over generations.

    Shows both best solution and population average fitness over time.

    Args:
        monitor: GAMonitor instance with collected data
        ga_instance: PyGAD GA instance (for final statistics)
        title: Plot title
        save_path: File path to save plot (optional)
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot best distance - purple (primary line)
    ax.plot(monitor.generation_numbers, monitor.best_distances,
            label='Best distance', linewidth=3, color=COLORS['primary'])

    # Plot average distance - yellow (shows population diversity)
    ax.plot(monitor.generation_numbers, monitor.avg_distances,
            label='Average population distance', linewidth=3,
            color=COLORS['accent'], alpha=0.9)

    final_distance = monitor.best_distances[-1]
    num_generations = ga_instance.generations_completed

    ax.set_xlabel('Generation', fontsize=16, fontweight='bold')
    ax.set_ylabel('Route Distance', fontsize=16, fontweight='bold')
    ax.set_title(f'{title}\n'
                 f'Best route: {final_distance:.2f} | '
                 f'Generations: {num_generations}',
                 fontsize=20, fontweight='bold', pad=20)
    ax.legend(fontsize=14, loc='best')
    ax.grid(True, alpha=0.3, linewidth=1.5)

    ax.tick_params(axis='both', which='major', labelsize=13)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def create_animation(tsp_problem, monitor, interval_generations=5, save_path='animation.gif'):
    """
    Create GIF animation showing route evolution across generations.

    Visualizes how the best solution improves over time, with all possible
    connections shown in the background for context.

    Args:
        tsp_problem: TSPProblem instance
        monitor: GAMonitor with recorded generation data
        interval_generations: Sample every N generations for smoother animation
        save_path: Output file path for GIF
    """
    print(f"Creating animation...")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Select frames to include (sample every N generations + final)
    frames_indices = list(range(0, len(monitor.best_routes), interval_generations))
    if len(monitor.best_routes) - 1 not in frames_indices:
        frames_indices.append(len(monitor.best_routes) - 1)

    def update(frame_idx):
        ax.clear()

        generation = monitor.generation_numbers[frame_idx]
        route = monitor.best_routes[frame_idx]
        distance = monitor.best_distances[frame_idx]

        # STEP 1: Draw all possible connections (background context)
        for i in range(tsp_problem.num_cities):
            for j in range(i + 1, tsp_problem.num_cities):
                x_coords = [tsp_problem.cities[i, 0], tsp_problem.cities[j, 0]]
                y_coords = [tsp_problem.cities[i, 1], tsp_problem.cities[j, 1]]
                ax.plot(x_coords, y_coords,
                        color=COLORS['connections'], linewidth=0.5, alpha=0.3, zorder=1)

        # STEP 2: Draw current best route (highlighted)
        route_cities = tsp_problem.cities[route.astype(int)]
        route_cities = np.vstack([route_cities, route_cities[0]])  # Close the loop

        ax.plot(route_cities[:, 0], route_cities[:, 1],
                color=COLORS['secondary'], linewidth=2.5, alpha=0.9, zorder=2,
                linestyle='--')

        # STEP 3: Draw city nodes
        ax.scatter(tsp_problem.cities[:, 0], tsp_problem.cities[:, 1],
                   c=COLORS['accent'], s=200, zorder=3,
                   edgecolors='white', linewidth=2)

        # STEP 4: Add city numbers
        for i, (x, y) in enumerate(tsp_problem.cities):
            ax.text(x, y, str(i + 1), ha='center', va='center',
                    fontsize=9, fontweight='bold', color='black', zorder=4)

        # STEP 5: Mark starting point
        ax.scatter(route_cities[0, 0], route_cities[0, 1],
                   c='red', s=300, marker='o', zorder=5,
                   edgecolors='darkred', linewidth=2)

        # Labels and formatting
        ax.set_title(f"Generation: {generation} | Distance: {distance:.2f}",
                     fontsize=18, fontweight='bold', pad=15)
        ax.set_xlabel('X Coordinate', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y Coordinate', fontsize=14, fontweight='bold')

        ax.grid(False)
        ax.set_aspect('equal')
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_facecolor('white')

    anim = FuncAnimation(fig, update, frames=frames_indices,
                         interval=500, repeat=True)

    writer = PillowWriter(fps=2)
    anim.save(save_path, writer=writer)
    print(f"Saved: {save_path}")

    plt.close()


def plot_route_with_all_connections(tsp_problem, route, title="TSP Best Route Using GA", save_path=None):
    """
    Visualize TSP route with all possible connections shown for context.

    Shows the complete graph with the optimal route highlighted, making it
    easy to see which connections were chosen vs. rejected.

    Args:
        tsp_problem: TSPProblem instance
        route: Array of city indices representing the solution
        title: Plot title
        save_path: File path to save plot (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # STEP 1: Draw all possible connections (light background)
    for i in range(tsp_problem.num_cities):
        for j in range(i + 1, tsp_problem.num_cities):
            x_coords = [tsp_problem.cities[i, 0], tsp_problem.cities[j, 0]]
            y_coords = [tsp_problem.cities[i, 1], tsp_problem.cities[j, 1]]
            ax.plot(x_coords, y_coords,
                    color=COLORS['connections'], linewidth=0.5, alpha=0.3, zorder=1)

    # STEP 2: Draw best route (thick highlighted line)
    route_cities = tsp_problem.cities[route.astype(int)]
    route_cities = np.vstack([route_cities, route_cities[0]])  # Close the loop

    ax.plot(route_cities[:, 0], route_cities[:, 1],
            color=COLORS['secondary'], linewidth=2.5, alpha=0.9, zorder=2,
            label='Best Route', linestyle='--')

    # STEP 3: Draw city nodes
    ax.scatter(tsp_problem.cities[:, 0], tsp_problem.cities[:, 1],
               c=COLORS['accent'], s=200, zorder=3, edgecolors='white', linewidth=2)

    # STEP 4: Add city numbers
    for i, (x, y) in enumerate(tsp_problem.cities):
        ax.text(x, y, str(i + 1), ha='center', va='center',
                fontsize=9, fontweight='bold', color='black', zorder=4)

    # STEP 5: Mark starting point
    ax.scatter(route_cities[0, 0], route_cities[0, 1],
               c='red', s=300, marker='o', zorder=5,
               edgecolors='darkred', linewidth=2)

    # Calculate total distance
    distance = tsp_problem.calculate_route_distance(route)

    # Labels and formatting
    ax.set_title(f"{title}\nTotal Distance: {distance:.2f}",
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('X Coordinate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=14, fontweight='bold')

    ax.legend(fontsize=12, loc='upper right', framealpha=0.9)

    ax.grid(False)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='major', labelsize=11)

    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results_to_csv(results, filename="experiment_results.csv"):
    """
    Append experiment results to CSV file.

    Creates file with headers if it doesn't exist. Appends to existing file
    to accumulate results from multiple runs.

    Args:
        results: List of dictionaries containing experiment results
        filename: Output CSV filename
    """
    from pathlib import Path

    file_exists = Path(filename).exists()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Open file in append mode
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        fieldnames = [
            'timestamp', 'run_id', 'experiment_name', 'description',
            'population_size', 'num_generations', 'mutation_percent_genes',
            'crossover_type', 'parent_selection_type', 'K_tournament',
            'best_distance', 'generations_completed'
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write header only if file doesn't exist
        if not file_exists:
            writer.writeheader()

        # Save each result
        for result in results:
            params = result['params']
            writer.writerow({
                'timestamp': timestamp,
                'run_id': result.get('run_id', 1),
                'experiment_name': result['name'],
                'description': result['description'],
                'population_size': params['population_size'],
                'num_generations': params['num_generations'],
                'mutation_percent_genes': params['mutation_percent_genes'],
                'crossover_type': params['crossover_type'],
                'parent_selection_type': params['parent_selection_type'],
                'K_tournament': params.get('K_tournament', ''),
                'best_distance': result['best_distance'],
                'generations_completed': result['generations']
            })

    print(f"\nResults saved to: {filename}")


# ============================================================================
# CUSTOM CROSSOVER OPERATORS FOR TSP
# ============================================================================

def crossover_func_ox(parents, offspring_size, ga_instance):
    """
    Ordered Crossover (OX) - specialized for TSP permutation problems.

    This crossover operator preserves the relative order and position of cities
    from parents while avoiding duplicates. It's specifically designed for
    permutation problems like TSP where each gene (city) must appear exactly once.

    Algorithm:
    1. Select two random crossover points
    2. Copy segment between points from parent1
    3. Fill remaining positions with cities from parent2 in their original order

    Args:
        parents: Parent solutions selected for mating
        offspring_size: Tuple (number of offspring, number of genes)
        ga_instance: PyGAD instance

    Returns:
        Numpy array of offspring solutions
    """
    offspring = []
    num_genes = offspring_size[1]

    for _ in range(offspring_size[0]):
        # Select two parents randomly
        parent1_idx = np.random.randint(0, parents.shape[0])
        parent2_idx = np.random.randint(0, parents.shape[0])
        parent1 = parents[parent1_idx].copy()
        parent2 = parents[parent2_idx].copy()

        # Select two random crossover points
        point1 = np.random.randint(0, num_genes - 1)
        point2 = np.random.randint(point1 + 1, num_genes)

        # Create offspring with placeholder values
        child = np.full(num_genes, -1, dtype=int)

        # Copy segment from parent1 between crossover points
        child[point1:point2] = parent1[point1:point2]

        # Fill remaining positions from parent2 (maintaining order)
        parent2_filtered = [gene for gene in parent2 if gene not in child[point1:point2]]

        current_pos = point2
        for gene in parent2_filtered:
            child[current_pos % num_genes] = gene
            current_pos += 1

        offspring.append(child)

    return np.array(offspring)