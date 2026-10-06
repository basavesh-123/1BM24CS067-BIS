import numpy as np
# 1. Define the Agricultural Monitoring Environment (Fitness Function)
def evaluate_irrigation_system(params):
    """
    Simulates an IoT-based irrigation system.
    Optimizes PID parameters: Kp (params[0]), Ki (params[1]), Kd (params[2]).
    Goal: Minimize the error from ideal soil moisture, water waste, and pump energy usage.
    """
    Kp, Ki, Kd = params
    
    # Mock simulation variables representing a day of moisture tracking
    time_steps = 100
    target_moisture = 60.0  # Ideal soil moisture percentage
    current_moisture = 35.0 # Starting dry moisture percentage
    
    total_error = 0.0
    water_pumped = 0.0
    integral_error = 0.0
    prev_error = 0.0
    
    for t in range(time_steps):
        error = target_moisture - current_moisture
        integral_error += error
        derivative_error = error - prev_error
        
        # Calculate pump action based on the PID parameters
        pump_output = (Kp * error) + (Ki * integral_error) + (Kd * derivative_error)
        pump_output = np.clip(pump_output, 0, 100)  # Pump capacity limit (0% to 100%)
        
        # Environmental impact (e.g., absorption, natural evaporation)
        moisture_gain = pump_output * 0.15 
        evaporation = 0.5 
        
        current_moisture += moisture_gain - evaporation
        current_moisture = np.clip(current_moisture, 0, 100)
        
        # Penalize variance from target, excess water pumping, and high energy usage
        total_error += abs(error)
        water_pumped += pump_output
        prev_error = error

    # The Cost Function: We want to minimize this score
    fitness_score = (total_error * 2.0) + (water_pumped * 0.5)
    return fitness_score

# 2. Define the Particle Structure
class Particle:
    def __init__(self, bounds):
        self.bounds = bounds
        # Initialize position randomly within bounds: [Kp, Ki, Kd]
        self.position = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        # Initialize velocity to zeros
        self.velocity = np.zeros(len(bounds))
        # Evaluate initial fitness
        self.fitness = evaluate_irrigation_system(self.position)
        # Store personal best performance
        self.best_position = np.copy(self.position)
        self.best_fitness = self.fitness

    def update_position(self):
        self.position += self.velocity
        # Ensure particle stays within allowed parameter limits
        for i in range(len(self.bounds)):
            self.position[i] = np.clip(self.position[i], self.bounds[i][0], self.bounds[i][1])
        # Re-evaluate fitness
        self.fitness = evaluate_irrigation_system(self.position)

# 3. The Particle Swarm Optimization Loop
def particle_swarm_optimization(num_particles=30, max_iter=50):
    # Search boundaries for PID gains: [(min_Kp, max_Kp), (min_Ki, max_Ki), (min_Kd, max_Kd)]
    bounds = [(0.0, 10.0), (0.0, 5.0), (0.0, 5.0)]
    
    # Hyperparameters for swarm intelligence
    w = 0.5   # Inertia weight (momentum)
    c1 = 1.5  # Cognitive coefficient (personal best influence)
    c2 = 1.5  # Social coefficient (global best influence)
    
    # Initialize the swarm population
    swarm = [Particle(bounds) for _ in range(num_particles)]
    
    # Identify initial global best particle
    global_best_particle = min(swarm, key=lambda p: p.best_fitness)
    global_best_position = np.copy(global_best_particle.best_position)
    global_best_fitness = global_best_particle.best_fitness
    
    print(f"Initial Best Fitness: {global_best_fitness:.2f}")

    # Optimization iterations
    for iteration in range(max_iter):
        for particle in swarm:
            # Generate stochastic weights
            r1, r2 = np.random.rand(), np.random.rand()
            
            # Calculate new velocity vectors
            cognitive_velocity = c1 * r1 * (particle.best_position - particle.position)
            social_velocity = c2 * r2 * (global_best_position - particle.position)
            particle.velocity = (w * particle.velocity) + cognitive_velocity + social_velocity
            
            # Apply displacement and calculate new fitness
            particle.update_position()
            
            # Evaluate against Personal Best
            if particle.fitness < particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_position = np.copy(particle.position)
                
                # Evaluate against Global Best
                if particle.fitness < global_best_fitness:
                    global_best_fitness = particle.fitness
                    global_best_position = np.copy(particle.position)
                    
        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"Iteration {iteration+1:02d}/{max_iter} | Best Cost: {global_best_fitness:.2f}")
            
    return global_best_position, global_best_fitness

# 4. Execute optimization
if __name__ == "__main__":
    np.random.seed(42) # Replicable simulation states
    
    best_pid_gains, minimum_cost = particle_swarm_optimization()
    
    print("\n--- Optimization Complete ---")
    print(f"Optimal Kp (Proportional Gain): {best_pid_gains[0]:.4f}")
    print(f"Optimal Ki (Integral Gain):     {best_pid_gains[1]:.4f}")
    print(f"Optimal Kd (Derivative Gain):   {best_pid_gains[2]:.4f}")
    print(f"Optimized System Penalty Score: {minimum_cost:.2f}")
