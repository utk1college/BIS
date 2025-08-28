import random

# ------------------------------
# Problem Setup
# ------------------------------
jobs = [3, 2, 7, 5, 9, 4]  # processing times of jobs
num_jobs = len(jobs)
population_size = 20
generations = 100
crossover_rate = 0.8
mutation_rate = 0.2

# ------------------------------
# Fitness Function (Makespan)
# ------------------------------
def fitness(chromosome):
    time = 0
    for job in chromosome:
        time += jobs[job]
    return 1 / time   # smaller time → higher fitness

# ------------------------------
# Create Initial Population
# ------------------------------
def initial_population():
    population = []
    for _ in range(population_size):
        chromosome = list(range(num_jobs))
        random.shuffle(chromosome)
        population.append(chromosome)
    return population

# ------------------------------
# Selection (Tournament)
# ------------------------------
def selection(population):
    contenders = random.sample(population, 3)
    contenders.sort(key=lambda chromo: fitness(chromo), reverse=True)
    return contenders[0]

# ------------------------------
# Crossover (Order Crossover)
# ------------------------------
def crossover(p1, p2):
    if random.random() < crossover_rate:
        a, b = sorted(random.sample(range(num_jobs), 2))
        child = [-1] * num_jobs
        child[a:b] = p1[a:b]
        fill = [x for x in p2 if x not in child]
        j = 0
        for i in range(num_jobs):
            if child[i] == -1:
                child[i] = fill[j]
                j += 1
        return child
    return p1[:]  # no crossover → copy parent

# ------------------------------
# Mutation (Swap)
# ------------------------------
def mutate(chromosome):
    if random.random() < mutation_rate:
        a, b = random.sample(range(num_jobs), 2)
        chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
    return chromosome

# ------------------------------
# Main GA Loop
# ------------------------------
population = initial_population()
best_solution = None
best_fit = -1

for gen in range(generations):
    new_pop = []
    for _ in range(population_size):
        parent1 = selection(population)
        parent2 = selection(population)
        child = crossover(parent1, parent2)
        child = mutate(child)
        new_pop.append(child)
   
    population = new_pop
   
    # Track best
    for chromo in population:
        fit = fitness(chromo)
        if fit > best_fit:
            best_fit = fit
            best_solution = chromo

# ------------------------------
# Result
# ------------------------------
print("Best Job Order:", best_solution)
print("Job Times:", [jobs[j] for j in best_solution])
print("Total Completion Time (Makespan):", sum(jobs[j] for j in best_solution))
