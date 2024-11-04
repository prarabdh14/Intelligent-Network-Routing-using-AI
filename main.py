import numpy as np
import random
import matplotlib.pyplot as plt

# Parameters
NUM_STATES = 10        # Number of network nodes
NUM_ACTIONS = 10       # Actions (next hops)
ALPHA = 0.1            # Learning rate
GAMMA = 0.9            # Discount factor
EPSILON = 0.1          # Exploration rate for Q-learning
NUM_EPISODES = 1000    # Training episodes for Q-learning
TRAFFIC_PATTERNS = 20  # Traffic samples to simulate

# Initialize Q-table
Q_table = np.zeros((NUM_STATES, NUM_ACTIONS))

# Generate network traffic (data collection)
def simulate_network_data():
    traffic_data = []
    for _ in range(TRAFFIC_PATTERNS):
        src, dest = random.randint(0, NUM_STATES-1), random.randint(0, NUM_STATES-1)
        latency = random.uniform(1, 10)  # Simulated latency between 1-10 ms
        throughput = random.uniform(10, 100)  # Throughput in Mbps
        packet_loss = random.uniform(0, 0.1)  # Packet loss between 0-10%
        traffic_data.append((src, dest, latency, throughput, packet_loss))
    return traffic_data

# Define reward function based on latency and packet loss
def calculate_reward(latency, packet_loss):
    return 1 / (latency + packet_loss * 100)

# Q-learning update rule
def q_learning_update(state, action, reward, next_state):
    best_next_action = np.argmax(Q_table[next_state])
    Q_table[state, action] += ALPHA * (reward + GAMMA * Q_table[next_state, best_next_action] - Q_table[state, action])

# Training loop
def train_q_learning():
    traffic_data = simulate_network_data()
    for episode in range(NUM_EPISODES):
        state = random.randint(0, NUM_STATES-1)
        for src, dest, latency, throughput, packet_loss in traffic_data:
            if np.random.uniform(0, 1) < EPSILON:
                action = random.randint(0, NUM_ACTIONS-1)  # Explore
            else:
                action = np.argmax(Q_table[state])  # Exploit
            
            reward = calculate_reward(latency, packet_loss)
            next_state = (state + 1) % NUM_STATES  # Simulated next state
            q_learning_update(state, action, reward, next_state)
            state = next_state

# Genetic Algorithm for Path Optimization
def genetic_algorithm():
    population_size = 10
    num_generations = 50
    mutation_rate = 0.1

    # Generate initial population of random paths
    population = [random.sample(range(NUM_STATES), NUM_STATES) for _ in range(population_size)]

    def fitness(route):
        return sum(Q_table[route[i], route[i + 1]] for i in range(len(route) - 1))

    for generation in range(num_generations):
        population = sorted(population, key=fitness, reverse=True)
        new_population = population[:2]  # Keep top 2 paths

        # Crossover
        for _ in range(population_size // 2 - 1):
            parent1, parent2 = random.sample(population[:5], 2)
            start, end = sorted(random.sample(range(NUM_STATES), 2))
            child = parent1[:start] + parent2[start:end] + parent1[end:]
            new_population.append(child)

        # Mutation
        for individual in new_population:
            if random.uniform(0, 1) < mutation_rate:
                i, j = random.sample(range(NUM_STATES), 2)
                individual[i], individual[j] = individual[j], individual[i]

        population = new_population

    best_route = max(population, key=fitness)
    print("Best route found by Genetic Algorithm:", best_route)

# Ant Colony Optimization (ACO) for Shortest Path
def ant_colony_optimization():
    pheromone = np.ones((NUM_STATES, NUM_STATES))
    alpha = 1.0
    beta = 2.0
    decay = 0.5
    num_ants = 5
    num_iterations = 50

    def next_node_probability(current, unvisited):
        tau = pheromone[current, unvisited] ** alpha
        eta = (1 / (Q_table[current, unvisited] + 1)) ** beta
        prob = tau * eta
        return prob / prob.sum()

    for iteration in range(num_iterations):
        all_paths = []
        for ant in range(num_ants):
            path = [random.randint(0, NUM_STATES-1)]
            unvisited = list(set(range(NUM_STATES)) - {path[0]})
            while unvisited:
                probs = next_node_probability(path[-1], unvisited)
                next_node = np.random.choice(unvisited, p=probs)
                path.append(next_node)
                unvisited.remove(next_node)
            all_paths.append(path)

        # Pheromone update
        pheromone *= (1 - decay)
        for path in all_paths:
            for i in range(len(path) - 1):
                pheromone[path[i], path[i+1]] += 1.0 / Q_table[path[i], path[i+1]]

    print("Pheromone levels after ACO:", pheromone)

# Evaluation and Performance Assessment
def evaluate_performance():
    print("Q-table after training:\n", Q_table)
    avg_reward = np.mean([calculate_reward(random.uniform(1, 10), random.uniform(0, 0.1)) for _ in range(TRAFFIC_PATTERNS)])
    print("Average Reward (Lower latency and packet loss is better):", avg_reward)

# Main function to run the components
def main():
    print("Starting AI-driven Network Routing System")
    
    # Step 1: Train Q-Learning Model
    train_q_learning()
    
    # Step 2: Genetic Algorithm for Route Optimization
    genetic_algorithm()
    
    # Step 3: Ant Colony Optimization for Path Reinforcement
    ant_colony_optimization()
    
    # Step 4: Evaluate Performance
    evaluate_performance()
    
    # Visualization of Q-table (Optional)
    plt.imshow(Q_table, cmap="viridis")
    plt.colorbar()
    plt.title("Q-table (State-Action values)")
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.show()

if __name__ == "__main__":
    main()
