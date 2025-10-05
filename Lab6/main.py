import numpy as np
import random

# === Grid setup ===
GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)
OBSTACLES = [(2, i) for i in range(1, 4)]  # Vertical wall in column 2, rows 1 to 3

# === Parameters ===
POP_SIZE = 10
MAX_ITER = 50
PATH_LENGTH = 20  # fewer steps needed for small grid

# === Helper Functions ===

def is_valid(pos):
    x, y = pos
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and pos not in OBSTACLES

def move_toward_goal(current):
    moves = [(0,1), (1,0), (0,-1), (-1,0)]
    random.shuffle(moves)
    cx, cy = current
    gx, gy = GOAL
    moves.sort(key=lambda m: abs((cx + m[0]) - gx) + abs((cy + m[1]) - gy))
    for dx, dy in moves:
        new_pos = (cx + dx, cy + dy)
        if is_valid(new_pos):
            return new_pos
    return current

def generate_random_path():
    path = [START]
    visited = set(path)
    current = START
    for _ in range(PATH_LENGTH):
        current = move_toward_goal(current)
        if current in visited:
            continue
        path.append(current)
        visited.add(current)
        if current == GOAL:
            break
    return path

def path_cost(path):
    cost = len(path)
    if path[-1] != GOAL:
        dist = abs(path[-1][0] - GOAL[0]) + abs(path[-1][1] - GOAL[1])
        cost += 100 + dist
    for pos in path:
        if pos in OBSTACLES:
            cost += 50
    return cost

# === GWO Optimization ===

def gwo_optimize():
    wolves = [generate_random_path() for _ in range(POP_SIZE)]

    for iteration in range(MAX_ITER):
        wolves.sort(key=path_cost)
        alpha, beta, delta = wolves[0], wolves[1], wolves[2]
        a = 2 - iteration * (2 / MAX_ITER)

        for i in range(3, POP_SIZE):
            new_path = []
            for j in range(min(len(alpha), len(wolves[i]), PATH_LENGTH)):
                A = 2 * a * random.random() - a
                C = 2 * random.random()
                x_alpha = np.array(alpha[j])
                x_wolf = np.array(wolves[i][j])
                D_alpha = abs(C * x_alpha - x_wolf)
                X1 = x_alpha - A * D_alpha

                A = 2 * a * random.random() - a
                C = 2 * random.random()
                x_beta = np.array(beta[j])
                D_beta = abs(C * x_beta - x_wolf)
                X2 = x_beta - A * D_beta

                A = 2 * a * random.random() - a
                C = 2 * random.random()
                x_delta = np.array(delta[j])
                D_delta = abs(C * x_delta - x_wolf)
                X3 = x_delta - A * D_delta

                X_new = (X1 + X2 + X3) / 3
                X_new = tuple(map(int, np.clip(np.round(X_new), 0, GRID_SIZE - 1)))

                if is_valid(X_new):
                    new_path.append(X_new)
                else:
                    if new_path:
                        new_path.append(move_toward_goal(new_path[-1]))
                    else:
                        new_path.append(move_toward_goal(START))
            wolves[i] = new_path

    best_path = sorted(wolves, key=path_cost)[0]
    return best_path

# === Textual Output ===

def print_grid(path):
    grid = [["." for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    for x, y in OBSTACLES:
        grid[y][x] = "#"  # Obstacle

    for x, y in path:
        if (x, y) != START and (x, y) != GOAL and grid[y][x] != "#":
            grid[y][x] = "*"

    sx, sy = START
    gx, gy = GOAL
    grid[sy][sx] = "S"
    grid[gy][gx] = "G"

    print("\n=== GWO Path Grid ===")
    for row in grid:
        print(" ".join(row))
    
    print("\nBest Path (coordinates):")
    print(path)

    print(f"\nPath Length: {len(path)}")
    print(f"Cost: {path_cost(path)}")

# === Run ===

best = gwo_optimize()
print_grid(best)
