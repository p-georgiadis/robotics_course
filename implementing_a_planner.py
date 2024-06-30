import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from collections import defaultdict
import heapq
import math

# Define the 8-neighborhood deltas for movement
deltas = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (-1, -1), (1, -1), (-1, 1)
]
costs = [1, 1, 1, 1, math.sqrt(2), math.sqrt(2), math.sqrt(2), math.sqrt(2)]

def get_neighbors(node, grid_map):
    """Return the list of navigable neighbors of a given node along with their costs."""
    neighbors = []
    for delta, cost in zip(deltas, costs):
        new_node = (node[0] + delta[0], node[1] + delta[1])
        if 0 <= new_node[0] < len(grid_map) and 0 <= new_node[1] < len(grid_map[0]):
            if grid_map[new_node[0]][new_node[1]] <= 0.3:  # Navigable threshold
                neighbors.append((cost, new_node))
    return neighbors

def heuristic(node, goal):
    """Calculate the Euclidean distance heuristic from the current node to the goal."""
    return np.sqrt((goal[0] - node[0]) ** 2 + (goal[1] - node[1]) ** 2)

def a_star(start, goal, grid_map):
    """Perform A* algorithm to find the shortest path from start to goal in a grid."""
    distances = defaultdict(lambda: float('inf'))
    distances[start] = 0
    priority_queue = [(0, start)]
    visited = set()
    parent = {}

    plt.ion()
    plt.imshow(grid_map, cmap='gray', origin='upper')
    plt.plot(start[1], start[0], 'y*')  # Mark start
    plt.plot(goal[1], goal[0], 'y*')  # Mark goal

    while priority_queue:
        current_priority, current_node = heapq.heappop(priority_queue)

        if current_node in visited:
            continue
        visited.add(current_node)

        plt.plot(current_node[1], current_node[0], 'g*')  # Mark visited node
        plt.show()
        plt.pause(0.000001)

        if current_node == goal:
            path = []
            while current_node in parent:
                path.append(current_node)
                current_node = parent[current_node]
            path.append(start)
            path.reverse()
            plt.ioff()
            for p in path:
                plt.plot(p[1], p[0], 'r.')  # Mark path
            plt.plot(start[1], start[0], 'y*')  # Ensure start is still marked
            plt.plot(goal[1], goal[0], 'y*')  # Ensure goal is still marked
            plt.show()
            return path

        for cost, neighbor in get_neighbors(current_node, grid_map):
            distance = distances[current_node] + cost
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                priority = distance + heuristic(neighbor, goal)
                heapq.heappush(priority_queue, (priority, neighbor))
                parent[neighbor] = current_node

    plt.ioff()
    plt.show()
    return None

def visualize_path(path, grid_map):
    """Visualize the path on the grid map."""
    map_copy = np.zeros((len(grid_map), len(grid_map[0]), 3))

    # Set colors: white for free space, black for obstacles, red for path
    for i in range(len(grid_map)):
        for j in range(len(grid_map[0])):
            if grid_map[i][j] > 0.3:  # Obstacle
                map_copy[i][j] = [0, 0, 0]  # Black
            else:  # Free space
                map_copy[i][j] = [1, 1, 1]  # White

    # Mark the path
    for node in path:
        map_copy[node[0]][node[1]] = [1, 0, 0]  # Red

    plt.imshow(map_copy)
    plt.title('Path Visualization')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.show()

def generate_random_map(rows, cols, occupancy=0.1):
    """Generate a random map with given dimensions and occupancy."""
    return np.random.rand(rows, cols) < occupancy

# Generate random map for testing
rows, cols = 20, 30
random_map = generate_random_map(rows, cols)

# Ensure random start and goal are in navigable spaces
start = (np.random.randint(rows), np.random.randint(cols))
goal = (np.random.randint(rows), np.random.randint(cols))
while random_map[start[0]][start[1]] or random_map[goal[0]][goal[1]]:
    start = (np.random.randint(rows), np.random.randint(cols))
    goal = (np.random.randint(rows), np.random.randint(cols))

# Perform A* on random map and visualize
path = a_star(start, goal, random_map)
if path:
    print("Random map path found:", path)
    visualize_path(path, random_map)
else:
    print("No path found on random map.")
