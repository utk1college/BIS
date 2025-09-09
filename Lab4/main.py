import numpy as np
import random

# Coordinates of depot + customers (0 is depot)
coords = np.array([
    [40, 50],  # depot
    [45, 68], [50, 30], [55, 20], [60, 80], [65, 60], [70, 40]
])

num_vehicles = 2
num_ants = 10
num_iterations = 100
alpha = 1.0  # pheromone importance
beta = 5.0   # heuristic importance (inverse distance)
rho = 0.5    # pheromone evaporation rate
initial_pheromone = 1.0

num_cities = len(coords)

# Distance matrix
dist_matrix = np.sqrt(((coords[:, None] - coords[None, :])**2).sum(axis=2))

# Heuristic matrix (inverse distance), avoid division by zero
heuristic = 1 / (dist_matrix + np.diag([np.inf]*num_cities))

# Initialize pheromone trails
pheromone = np.ones((num_cities, num_cities)) * initial_pheromone

def choose_next_city(current_city, unvisited, pheromone, heuristic):
    pheromone_vals = pheromone[current_city][unvisited] ** alpha
    heuristic_vals = heuristic[current_city][unvisited] ** beta
    probs = pheromone_vals * heuristic_vals
    probs /= probs.sum()
    return np.random.choice(unvisited, p=probs)

def construct_solution():
    routes = [[] for _ in range(num_vehicles)]
    unvisited = set(range(1, num_cities))  # customers only
    for v in range(num_vehicles):
        routes[v].append(0)  # start from depot

    while unvisited:
        for v in range(num_vehicles):
            current_city = routes[v][-1]
            candidates = list(unvisited)
            if not candidates:
                break
            next_city = choose_next_city(current_city, candidates, pheromone, heuristic)
            routes[v].append(next_city)
            unvisited.remove(next_city)
            if not unvisited:
                break

    # Return to depot
    for v in range(num_vehicles):
        routes[v].append(0)
    return routes

def route_length(route):
    length = 0
    for i in range(len(route)-1):
        length += dist_matrix[route[i], route[i+1]]
    return length

best_routes = None
best_length = float('inf')

for iteration in range(num_iterations):
    all_routes = []
    all_lengths = []

    for _ in range(num_ants):
        routes = construct_solution()
        total_length = sum(route_length(r) for r in routes)
        all_routes.append(routes)
        all_lengths.append(total_length)

        if total_length < best_length:
            best_length = total_length
            best_routes = routes

    # Pheromone evaporation
    pheromone *= (1 - rho)

    # Pheromone update (only best ant deposits pheromone)
    for route in best_routes:
        for i in range(len(route)-1):
            from_city = route[i]
            to_city = route[i+1]
            pheromone[from_city][to_city] += 1 / best_length
            pheromone[to_city][from_city] += 1 / best_length

print("Best total route length:", best_length)
for v, route in enumerate(best_routes):
    print(f"Vehicle {v+1} route: {route}")
