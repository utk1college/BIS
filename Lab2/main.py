import random
import math

# ----------------------------
# Problem: TSP cities
# ----------------------------
cities = [(0,0), (1,5), (5,2), (6,6), (8,3)]  # coordinates
num_cities = len(cities)

# Parameters
population_size = 30
generations = 200
crossover_rate = 0.8
mutation_rate = 0.2

# ----------------------------
# Distance Function
# ----------------------------
def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def tour_length(chromosome):
    length = 0
    for i in range(num_cities):
        length += distance(cities[chromosome[i]], cities[chromosome[(i+1)%num_cities]])
    return length

# ----------------------------
# Fitness Function
# ----------------------------
def fitness(chromosome):
    return 1 / tour_length(chromosome)

# ----------------------------
# Population Initialization
# ----------------------------
def initial_population():
    population = []
    for _ in range(population_size):
        chromosome = list(range(num_cities))
        random.shuffle(chromosome)
        population.append(chromosome)
    return population

# ----------------------------
# Selection (Tournament)
# ----------------------------
def selection(population):
    contenders = random.sample(population, 3)
    contenders.sort(key=lambda c: fitness(c), reverse=True)
    return contenders[0]

# ----------------------------
# Crossover (Order Crossover OX)
# ----------------------------
def crossover(p1, p2):
    if random.random() < crossover_rate:
        a, b = sorted(random.sample(range(num_cities), 2))
        child = [-1]*num_cities
        child[a:b] = p1[a:b]
        fill = [x for x in p2 if x not in child]
        j = 0
        for i in range(num_cities):
            if child[i] == -1:
                child[i] = fill[j]
                j += 1
        return child
    return p1[:]

# ----------------------------
# Mutation (Swap)
# ----------------------------
def mutate(chromosome):
    if random.random() < mutation_rate:
        a, b = random.sample(range(num_cities), 2)
        chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
    return chromosome

# ----------------------------
# Main GEA Loop
# ----------------------------
population = initial_population()
best_solution = None
best_distance = float("inf")

for g in range(generations):
    new_pop = []
    for _ in range(population_size):
        parent1 = selection(population)
        parent2 = selection(population)
        child = crossover(parent1, parent2)
        child = mutate(child)
        new_pop.append(child)
   
    population = new_pop
   
    # Track best solution
    for chromo in population:
        d = tour_length(chromo)
        if d < best_distance:
            best_distance = d
            best_solution = chromo

# ----------------------------
# Result
# ----------------------------
print("Best Tour (order of cities):", best_solution)
print("Best Tour Distance:", best_distance)
