import numpy as np
import random

# ---------------- Knapsack Problem Setup ----------------
# Example items: (value, weight)
items = [(60, 10), (100, 20), (120, 30)]
capacity = 50
n = len(items)

def fitness(solution):
    total_value = total_weight = 0
    for i in range(n):
        if solution[i] == 1:
            total_value += items[i][0]
            total_weight += items[i][1]
    if total_weight > capacity:
        return 0  # invalid solution
    return total_value

# ---------------- Cuckoo Search Algorithm ----------------
def levy_flight(Lambda):
    u = np.random.normal(0, 1) * np.power(abs(np.random.normal(0, 1)), -1.0 / Lambda)
    v = np.random.normal(0, 1)
    step = u / abs(v) ** (1 / Lambda)
    return step

def get_random_solution():
    return [random.randint(0, 1) for _ in range(n)]

def cuckoo_search(num_nests=10, pa=0.25, max_iter=100):
    nests = [get_random_solution() for _ in range(num_nests)]
    best = max(nests, key=fitness)

    for _ in range(max_iter):
        # Generate new solution via Levy flight
        cuckoo = best[:]
        step = int(abs(round(levy_flight(1.5)))) % n
        pos = random.randint(0, n-1)
        cuckoo[pos] = 1 - cuckoo[pos]  # flip bit
        
        # Replace a random nest if better
        j = random.randint(0, num_nests-1)
        if fitness(cuckoo) > fitness(nests[j]):
            nests[j] = cuckoo
        
        # Abandon some nests with probability pa
        for i in range(num_nests):
            if random.random() < pa:
                nests[i] = get_random_solution()
        
        # Update best
        best = max(nests, key=fitness)
    
    return best, fitness(best)

# ---------------- Run the algorithm ----------------
solution, value = cuckoo_search()
print("Best solution:", solution)
print("Total value:", value)
