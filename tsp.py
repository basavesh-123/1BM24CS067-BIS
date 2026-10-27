import numpy as np

class AntColonyTSP:
    def __init__(self, distances, n_ants=10, n_best=2, n_iterations=100, decay=0.5, alpha=1, beta=2):
        """
        Args:
            distances (2D array): Square matrix of distances between cities.
            n_ants (int): Number of ants running per iteration.
            n_best (int): Number of best ants who deposit pheromones.
            n_iterations (int): Number of iterations.
            decay (float): Pheromone decay rate (evaporation).
            alpha (int/float): Pheromone importance exponent.
            beta (int/float): Distance importance exponent.
        """
        self.distances = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheromone(all_paths, self.n_best, shortest_path=shortest_path)
            shortest_path = min(all_paths, key=lambda x: x[1])
            
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            
            # Evaporate pheromones
            self.pheromone *=(1 - self.decay)            
            
        return all_time_shortest_path

    def spread_pheromone(self, all_paths, n_best, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]

    def gen_path_dist(self, path):
        total_dist = 0
        for ele in path:
            total_dist += self.distances[ele]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start)) # Back to the starting city
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0

        # Avoid division by zero for identical city nodes or unvisited setups
        row = (pheromone ** self.alpha) * ((1.0 / (dist + 1e-10)) ** self.beta)

        norm_row = row / row.sum()
        move = np.random.choice(self.all_inds, p=norm_row)
        return move

# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    # Generate 5 coordinates representing 5 cities (X, Y)
    np.random.seed(42)
    cities = np.random.rand(5, 2) * 100
    
    # Calculate the pairwise distance matrix between cities
    num_cities = len(cities)
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distance_matrix[i][j] = np.linalg.norm(cities[i] - cities[j])
            else:
                distance_matrix[i][j] = np.inf # Prevent choosing the same city

    # Initialize and run ACO
    ant_colony = AntColonyTSP(
        distances=distance_matrix, 
        n_ants=15, 
        n_best=3, 
        n_iterations=50, 
        decay=0.4, 
        alpha=1, 
        beta=3
    )
    
    best_path, best_distance = ant_colony.run()
    
    # Format path output nicely
    formatted_path = [edge[0] for edge in best_path] + [best_path[-1][1]]
    
    print(f"Best Path Found: {formatted_path}")
    print(f"Shortest Distance Found: {best_distance:.2f}")
