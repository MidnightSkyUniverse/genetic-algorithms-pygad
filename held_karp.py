import math
from itertools import combinations

# Your 20 city coordinates
cities = [
    [37.45, 95.07],
    [73.20, 59.86],
    [15.60, 15.60],
    [15.81, 86.62],
    [60.11, 70.81],
    [83.24, 43.19],
    [27.67, 52.92],
    [91.32, 18.47],
    [44.68, 30.15],
    [68.99, 89.23],
    [22.14, 67.45],
    [55.33, 12.78],
    [8.92, 38.51],
    [76.84, 24.66],
    [49.21, 77.38],
    [52.18, 43.92],
    [82.45, 68.31],
    [28.76, 28.54],
    [65.89, 15.77],
    [41.23, 60.45],
]

# Step 1: compute distance matrix
n = len(cities)
dist = [[0]*n for _ in range(n)]
for i in range(n):
    for j in range(n):
        dx = cities[i][0] - cities[j][0]
        dy = cities[i][1] - cities[j][1]
        dist[i][j] = math.hypot(dx, dy)

# Step 2: Held–Karp dynamic programming
# dp[(S, i)] = (min_distance, previous_city)
dp = {}
for k in range(1, n):
    dp[(1 << k, k)] = (dist[0][k], 0)

for subset_size in range(2, n):
    for subset in combinations(range(1, n), subset_size):
        bits = 0
        for bit in subset:
            bits |= 1 << bit
        for k in subset:
            prev_bits = bits & ~(1 << k)
            res = []
            for m in subset:
                if m == k:
                    continue
                if (prev_bits, m) in dp:
                    res.append((dp[(prev_bits, m)][0] + dist[m][k], m))
            dp[(bits, k)] = min(res)

# Step 3: close the tour (return to start)
bits = (1 << n) - 2
res = []
for k in range(1, n):
    res.append((dp[(bits, k)][0] + dist[k][0], k))
opt_cost, parent = min(res)

# Step 4: reconstruct path
path = [0]
mask = (1 << n) - 2
last = parent
for i in range(n - 1):
    path.append(last)
    new_mask = mask & ~(1 << last)
    _, last = dp[(mask, last)]
    mask = new_mask
path.append(0)
path = path[::-1]

# Output
print(f"Optimal route (0-based indices): {path}")
print(f"Optimal route length: {opt_cost:.4f}")
