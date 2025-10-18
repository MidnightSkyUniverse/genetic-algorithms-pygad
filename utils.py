"""
utils.py - Funkcje pomocnicze i klasy dla TSP z algorytmem genetycznym
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
# KLASA TSP PROBLEM
# ============================================================================

class TSPProblem:
    """Klasa reprezentująca problem komiwojażera"""
    
    def __init__(self, num_cities=15, fixed_cities=None, seed=42):
        """
        Inicjalizacja problemu TSP
        
        Args:
            num_cities: Liczba miast
            fixed_cities: Numpy array ze współrzędnymi ustalonych miast
            seed: Ziarno losowości jeśli generujemy losowe miasta
        """
        self.num_cities = num_cities
        
        if fixed_cities is not None:
            self.cities = np.array(fixed_cities)
        else:
            np.random.seed(seed)
            self.cities = np.random.rand(num_cities, 2) * 100
        
        self.distance_matrix = self._calculate_distance_matrix()
        
    def _calculate_distance_matrix(self):
        """Oblicz macierz odległości między wszystkimi miastami"""
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
        """Oblicz całkowitą długość trasy"""
        distance = 0
        for i in range(len(route) - 1):
            distance += self.distance_matrix[int(route[i]), int(route[i+1])]
        distance += self.distance_matrix[int(route[-1]), int(route[0])]
        return distance


# ============================================================================
# MONITORING DANYCH
# ============================================================================

class GAMonitor:
    """Klasa do zbierania danych podczas działania AG"""
    
    def __init__(self):
        self.generation_numbers = []
        self.best_distances = []
        self.avg_distances = []
        self.best_routes = []
        
    def reset(self):
        """Resetuj dane"""
        self.generation_numbers = []
        self.best_distances = []
        self.avg_distances = []
        self.best_routes = []

    def record_generation(self, ga_instance, tsp_problem):
        """Zapisz dane z generacji"""

        distances = []
        for solution in ga_instance.population:
            dist = tsp_problem.calculate_route_distance(solution)
            distances.append(dist)
            if len(solution) != len(set(solution)):
                print(f"⚠️ Niepoprawne rozwiązanie w gen {ga_instance.generations_completed}: {solution}")

        # Znajdź minimum
        best_idx = np.argmin(distances)
        best_distance = distances[best_idx]
        best_solution = ga_instance.population[best_idx].copy()
        avg_distance = np.mean(distances)

        population_distances = [
            tsp_problem.calculate_route_distance(sol)
            for sol in ga_instance.population
        ]
        avg_distance = np.mean(population_distances)

        self.generation_numbers.append(ga_instance.generations_completed)
        self.best_distances.append(best_distance)
        self.avg_distances.append(avg_distance)
        self.best_routes.append(solution.copy())

        # if ga_instance.generations_completed % 10 == 0:
        #     print(f"Generacja {ga_instance.generations_completed:3d} | "
        #           f"Najlepsza: {best_distance:.2f} | Średnia: {avg_distance:.2f}")


# ============================================================================
# FUNKCJE POMOCNICZE
# ============================================================================

def load_config(config_path="config.yaml"):
    """Załaduj konfigurację z pliku YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_output_dirs(base_dir="outputs"):
    """Stwórz foldery na outputy"""
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
    """Połącz parametry baseline z parametrami eksperymentu"""
    params = baseline.copy()
    params.update(experiment_params)
    return params


# ============================================================================
# URUCHOMIENIE AG
# ============================================================================

def run_tsp_ga(tsp_problem, monitor, params):
    """
    Uruchom algorytm genetyczny dla TSP
    
    Args:
        tsp_problem: Instancja TSPProblem
        monitor: Instancja GAMonitor
        params: Słownik z parametrami AG
        
    Returns:
        Instancja pygad.GA po zakończeniu
    """
    
    monitor.reset()
    
    # Funkcja fitness - używa globalnego tsp_problem
    def fitness_func(ga_instance, solution, solution_idx):
        distance = tsp_problem.calculate_route_distance(solution)
        return 1.0 / (distance + 1e-6)
    
    # Callback dla każdej generacji
    def on_generation(ga_instance):
        monitor.record_generation(ga_instance, tsp_problem)

    # Wybór operatora krzyżowania
    if params['crossover_type'] == 'ox':
        crossover_func = crossover_func_ox
    else:
        crossover_func = params['crossover_type']
    
    # Konfiguracja PyGAD
    ga_instance = pygad.GA(
        num_generations=params['num_generations'],
        num_parents_mating=params['num_parents_mating'],
        sol_per_pop=params['population_size'],
        num_genes=tsp_problem.num_cities,
        
        fitness_func=fitness_func,
        
        gene_type=int,
        gene_space=list(range(tsp_problem.num_cities)),
        allow_duplicate_genes=False,
        
        parent_selection_type=params['parent_selection_type'],
        K_tournament=params.get('K_tournament'),
        
        crossover_type=crossover_func,
        crossover_probability=params.get('crossover_probability'),
        
        mutation_type=params.get('mutation_type', 'swap'),
        mutation_percent_genes=params['mutation_percent_genes'],
        mutation_probability=params.get('mutation_probability'),
        
        keep_elitism=params.get('keep_elitism'),
        
        on_generation=on_generation,
        
        stop_criteria=params.get('stop_criteria', 'saturate_50'),

        save_best_solutions=True,
    )
    
    # print(f"\nUruchamiam AG dla TSP z {tsp_problem.num_cities} miastami")
    # print(f"Parametry: pop={params['population_size']}, "
    #       f"gen={params['num_generations']}, "
    #       f"mut={params['mutation_percent_genes']}%",
    #       f"mut_prob={params['mutation_probability']}",
    #       )

    
    ga_instance.run()

    best_distance = float('inf')
    for idx, sol in enumerate(ga_instance.best_solutions):
        dist = tsp_problem.calculate_route_distance(sol)
        if dist < best_distance:  # Porównanie float vs float ✓
            best_distance = dist
            best_solution = sol.copy()
            best_generation = idx

    
    print(f"Najlepsza trasa: {best_distance:.2f}")
    print(f"Wykonano {ga_instance.generations_completed} generacji")
    
    return ga_instance, best_solution, best_distance


# ============================================================================
# WIZUALIZACJE
# ============================================================================

# Paleta kolorów
COLORS = {
    'primary': '#5c26ff',  # Fioletowy primary
    'secondary': '#9c31ff',  # Fioletowy secondary
    'accent': '#ffba0a',  # Żółty akcent
    'start': '#00ff88',  # Zielony dla startu (opcjonalnie)
    'connections': '#E0E0E0',
}



def plot_convergence(monitor, ga_instance, title="Zbieżność AG", save_path=None):
    """Wizualizuj zbieżność algorytmu"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Linia najlepszej odległości - fioletowy (łatwiej rozróżnialny)
    ax.plot(monitor.generation_numbers, monitor.best_distances,
            label='Najlepsza odległość', linewidth=3, color=COLORS['primary'])

    # Linia średniej odległości - żółty (wyraźny kontrast)
    ax.plot(monitor.generation_numbers, monitor.avg_distances,
            label='Średnia odległość w populacji', linewidth=3,
            color=COLORS['accent'], alpha=0.9)

    final_distance = monitor.best_distances[-1]
    num_generations = ga_instance.generations_completed

    ax.set_xlabel('Generacja', fontsize=16, fontweight='bold')
    ax.set_ylabel('Długość trasy', fontsize=16, fontweight='bold')
    ax.set_title(f'{title}\n'
                 f'Najlepsza trasa: {final_distance:.2f} | '
                 f'Liczba generacji: {num_generations}',
                 fontsize=20, fontweight='bold', pad=20)
    ax.legend(fontsize=14, loc='best')
    ax.grid(True, alpha=0.3, linewidth=1.5)

    # Większe znaczniki na osiach
    ax.tick_params(axis='both', which='major', labelsize=13)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Zapisano: {save_path}")
    else:
        plt.show()

    plt.close()


def create_animation(tsp_problem, monitor, interval_generations=5, save_path='animation.gif'):
    """Stwórz animację ewolucji trasy z widocznymi wszystkimi połączeniami"""
    print(f"Tworzę animację...")

    fig, ax = plt.subplots(figsize=(12, 10))


    frames_indices = list(range(0, len(monitor.best_routes), interval_generations))
    if len(monitor.best_routes) - 1 not in frames_indices:
        frames_indices.append(len(monitor.best_routes) - 1)

    def update(frame_idx):
        ax.clear()

        generation = monitor.generation_numbers[frame_idx]
        route = monitor.best_routes[frame_idx]
        distance = monitor.best_distances[frame_idx]

        # KROK 1: Rysuj wszystkie możliwe połączenia (tło)
        for i in range(tsp_problem.num_cities):
            for j in range(i + 1, tsp_problem.num_cities):
                x_coords = [tsp_problem.cities[i, 0], tsp_problem.cities[j, 0]]
                y_coords = [tsp_problem.cities[i, 1], tsp_problem.cities[j, 1]]
                ax.plot(x_coords, y_coords,
                        color=COLORS['connections'], linewidth=0.5, alpha=0.3, zorder=1)

        # KROK 2: Rysuj aktualną najlepszą trasę
        route_cities = tsp_problem.cities[route.astype(int)]
        route_cities = np.vstack([route_cities, route_cities[0]])

        ax.plot(route_cities[:, 0], route_cities[:, 1],
                color=COLORS['secondary'], linewidth=2.5, alpha=0.9, zorder=2,
                linestyle='--')

        # KROK 3: Rysuj miasta
        ax.scatter(tsp_problem.cities[:, 0], tsp_problem.cities[:, 1],
                   c=COLORS['accent'], s=200, zorder=3,
                   edgecolors='white', linewidth=2)

        # KROK 4: Numery miast
        for i, (x, y) in enumerate(tsp_problem.cities):
            ax.text(x, y, str(i + 1), ha='center', va='center',
                    fontsize=9, fontweight='bold', color='black', zorder=4)

        # KROK 5: Punkt startowy
        ax.scatter(route_cities[0, 0], route_cities[0, 1],
                   c='red', s=300, marker='o', zorder=5,
                   edgecolors='darkred', linewidth=2)

        # Tytuł i etykiety
        ax.set_title(f"Generation: {generation} | Distance: {distance:.2f}",
                     fontsize=18, fontweight='bold', pad=15)
        ax.set_xlabel('X Coordinate', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y Coordinate', fontsize=14, fontweight='bold')

        # Usuń siatkę
        ax.grid(False)
        ax.set_aspect('equal')
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Białe tło
        ax.set_facecolor('white')

    anim = FuncAnimation(fig, update, frames=frames_indices,
                         interval=500, repeat=True)

    writer = PillowWriter(fps=2)
    anim.save(save_path, writer=writer)
    print(f"Zapisano: {save_path}")

    plt.close()


def plot_route_with_all_connections(tsp_problem, route, title="TSP Best Route Using GA", save_path=None):
    """Wizualizuj trasę TSP z widocznymi wszystkimi możliwymi połączeniami"""
    fig, ax = plt.subplots(figsize=(12, 10))


    # KROK 1: Rysuj wszystkie możliwe połączenia (bardzo cienkie, jasne)
    for i in range(tsp_problem.num_cities):
        for j in range(i + 1, tsp_problem.num_cities):
            x_coords = [tsp_problem.cities[i, 0], tsp_problem.cities[j, 0]]
            y_coords = [tsp_problem.cities[i, 1], tsp_problem.cities[j, 1]]
            ax.plot(x_coords, y_coords,
                    color=COLORS['connections'], linewidth=0.5, alpha=0.3, zorder=1)

    # KROK 2: Rysuj najlepszą trasę (gruba, ciemna linia)
    route_cities = tsp_problem.cities[route.astype(int)]
    route_cities = np.vstack([route_cities, route_cities[0]])  # Zamknij pętlę

    ax.plot(route_cities[:, 0], route_cities[:, 1],
            color=COLORS['secondary'], linewidth=2.5, alpha=0.9, zorder=2,
            label='Best Route', linestyle='--')

    # KROK 3: Rysuj miasta (punkty)
    ax.scatter(tsp_problem.cities[:, 0], tsp_problem.cities[:, 1],
               c=COLORS['accent'], s=200, zorder=3, edgecolors='white', linewidth=2)

    # KROK 4: Dodaj numery miast
    for i, (x, y) in enumerate(tsp_problem.cities):
        ax.text(x, y, str(i + 1), ha='center', va='center',
                fontsize=9, fontweight='bold', color='black', zorder=4)

    # KROK 5: Oznacz punkt startowy
    ax.scatter(route_cities[0, 0], route_cities[0, 1],
               c='red', s=300, marker='o', zorder=5,
               edgecolors='darkred', linewidth=2)

    # Oblicz długość trasy
    distance = tsp_problem.calculate_route_distance(route)

    # Tytuł i etykiety
    ax.set_title(f"{title}\nTotal Distance: {distance:.2f}",
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('X Coordinate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=14, fontweight='bold')

    # Legenda
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9)

    # Usuń siatkę, zostaw tylko osie
    ax.grid(False)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='major', labelsize=11)

    # Białe tło
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Zapisano: {save_path}")
    else:
        plt.show()

    plt.close()


# ============================================================================
# ZAPISZ WYNIKI
# ============================================================================

def save_results_to_csv(results, filename="experiment_results_pop200_mut5.csv"):
    """
    Zapisz wyniki eksperymentów do pliku CSV

    Args:
        results: Lista słowników z wynikami eksperymentów
        filename: Nazwa pliku CSV
    """
    from pathlib import Path

    file_exists = Path(filename).exists()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Otwórz plik w trybie append
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        fieldnames = [
            'timestamp', 'run_id', 'experiment_name', 'description',
            'population_size', 'num_generations', 'mutation_percent_genes',
            'crossover_type', 'parent_selection_type', 'K_tournament',
            'best_distance', 'generations_completed'
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Napisz header tylko jeśli plik nie istnieje
        if not file_exists:
            writer.writeheader()

        # Zapisz każdy wynik
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

    print(f"\nWyniki zapisane do: {filename}")


# ============================================================================
# CUSTOM CROSSOVER OPERATORS FOR TSP
# ============================================================================

def crossover_func_ox(parents, offspring_size, ga_instance):
    """
    Ordered Crossover (OX) - specialized for TSP

    Preserves the order and position of cities from parents while avoiding duplicates.
    Better than standard crossover for permutation problems like TSP.

    Args:
        parents: Parent solutions selected for mating
        offspring_size: Tuple (number of offspring, number of genes)
        ga_instance: PyGAD instance

    Returns:
        numpy array of offspring solutions
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

        # Create offspring
        child = np.full(num_genes, -1, dtype=int)

        # Copy segment from parent1
        child[point1:point2] = parent1[point1:point2]

        # Fill remaining positions from parent2 in order
        parent2_filtered = [gene for gene in parent2 if gene not in child[point1:point2]]

        current_pos = point2
        for gene in parent2_filtered:
            child[current_pos % num_genes] = gene
            current_pos += 1

        offspring.append(child)

    return np.array(offspring)
