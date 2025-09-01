import numpy as np

# ---------- Step 1: Define Problem (Portfolio Optimization) ----------
# Expected returns for 4 assets (example data)
returns = np.array([0.12, 0.18, 0.15, 0.10])

# Covariance matrix of returns (risk measure)
cov_matrix = np.array([
    [0.010, 0.002, 0.001, 0.003],
    [0.002, 0.030, 0.002, 0.004],
    [0.001, 0.002, 0.020, 0.002],
    [0.003, 0.004, 0.002, 0.025]
])

# Fitness function: Sharpe ratio (maximize return / risk)
def fitness(weights):
    weights = np.array(weights)
    portfolio_return = np.dot(weights, returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    if portfolio_risk == 0:  # avoid division by zero
        return -999
    return portfolio_return / portfolio_risk

# ---------- Step 2: Initialize PSO Parameters ----------
num_particles = 30
num_assets = len(returns)
iterations = 100

w = 0.7      # inertia weight
c1 = 1.5     # cognitive coefficient
c2 = 1.5     # social coefficient

# ---------- Step 3: Initialize Particles ----------
positions = np.random.dirichlet(np.ones(num_assets), size=num_particles)  # weights sum=1
velocities = np.random.rand(num_particles, num_assets) * 0.1

personal_best_positions = positions.copy()
personal_best_scores = np.array([fitness(p) for p in positions])

global_best_position = personal_best_positions[np.argmax(personal_best_scores)]
global_best_score = np.max(personal_best_scores)

# ---------- Step 4: Main Loop ----------
for _ in range(iterations):
    for i in range(num_particles):
        # Update velocity
        r1, r2 = np.random.rand(num_assets), np.random.rand(num_assets)
        velocities[i] = (w * velocities[i]
                         + c1 * r1 * (personal_best_positions[i] - positions[i])
                         + c2 * r2 * (global_best_position - positions[i]))

        # Update position (weights must be valid portfolio)
        positions[i] += velocities[i]
        positions[i] = np.maximum(positions[i], 0)     # no negative weights
        positions[i] /= np.sum(positions[i])           # normalize to sum=1

        # Evaluate fitness
        score = fitness(positions[i])

        # Update personal best
        if score > personal_best_scores[i]:
            personal_best_scores[i] = score
            personal_best_positions[i] = positions[i].copy()

        # Update global best
        if score > global_best_score:
            global_best_score = score
            global_best_position = positions[i].copy()

# ---------- Step 5: Output Result ----------
print("Optimal Portfolio Weights:", global_best_position)
print("Best Sharpe Ratio:", global_best_score)
