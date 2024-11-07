## NetRouteAI: AI-Driven Network Routing Optimization

NetRouteAI is an AI-powered network routing system designed to optimize packet routing in a simulated network environment. This project leverages a combination of Q-learning, Genetic Algorithm, and Ant Colony Optimization (ACO) to enhance routing decisions, improve network efficiency, and reduce latency and packet loss.

## Project Overview

Network routing can be a complex problem, especially in environments with variable latency, throughput, and packet loss. NetRouteAI uses reinforcement learning (Q-learning) to train a model for routing, while Genetic Algorithms and Ant Colony Optimization are applied to further optimize path selection and reinforce the best routes.

This combination of techniques results in a routing system that can dynamically adjust to changing network conditions, selecting optimal paths based on learned rewards and past experiences.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Algorithms and Approach](#algorithms-and-approach)
- [Future Work](#future-work)
- [License](#license)

## Features

- **Q-learning-based Route Optimization**: Uses reinforcement learning to train a Q-table for different routing decisions.
- **Genetic Algorithm (GA)**: Optimizes paths by evolving a population of routes.
- **Ant Colony Optimization (ACO)**: Reinforces best routes based on pheromone trails.
- **Network Simulation**: Simulates network traffic patterns with metrics like latency, throughput, and packet loss.
- **Flexible & Extendable**: Configurable parameters for learning rate, exploration rate, and traffic patterns.

## Installation

To get started, clone this repository and install the required dependencies.

```bash
git clone https://github.com/yourusername/SmartNet-Pathways.git
cd SmartNet-Pathways
pip install -r requirements.txt
```

### Requirements

- Python 3.6+
- NumPy
- Matplotlib

## Usage

To run the project, execute the main script:

```bash
python main.py
```

This will initiate the training process using Q-learning, followed by route optimization with Genetic Algorithm and Ant Colony Optimization.

## Project Structure

```plaintext
SmartNet-Pathways/
├── main.py               # Main entry point to execute the program
├── q_learning.py         # Contains functions for Q-learning training
├── genetic_algorithm.py   # Contains the Genetic Algorithm implementation
├── ant_colony.py         # Contains the Ant Colony Optimization implementation
├── utils.py              # Helper functions for network simulation and reward calculation
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Algorithms and Approach

1. **Q-learning**:  
   Q-learning is a reinforcement learning algorithm used to train an agent by updating the Q-table based on the agent's interactions with the environment. The Q-table stores state-action values, and the model uses this table to choose actions that maximize future rewards based on latency and packet loss.

2. **Genetic Algorithm (GA)**:  
   GA is used for path optimization by evolving a population of paths. It selects the best paths based on fitness scores, performs crossover between high-fitness individuals, and applies mutations to introduce variability. This approach helps in finding efficient paths across the network.

3. **Ant Colony Optimization (ACO)**:  
   ACO reinforces optimal paths by simulating pheromone trails. Paths that are selected more frequently by virtual ants receive higher pheromone levels, which makes them more likely to be chosen in future iterations. This helps in refining the paths discovered by the Q-learning and GA stages.

4. **Reward Function**:  
   The reward function is based on latency and packet loss. Lower latency and lower packet loss result in higher rewards, which guides the learning process in Q-learning and path selection in ACO and GA.

## Example Output

Upon running the script, you should see:

- A trained Q-table visualized as a heatmap, showing the learned state-action values.
- Output indicating the best route found by the Genetic Algorithm.
- A final pheromone table that shows the reinforcement from Ant Colony Optimization.

## Future Work

- **Extend Simulation Metrics**: Integrate additional metrics like jitter and hop count for a more realistic simulation.
- **Dynamic Traffic Patterns**: Allow real-time traffic changes and see how the algorithms adapt.
- **Implement More Sophisticated Reward Functions**: Experiment with multi-objective rewards combining multiple network quality factors.
- **Enhance ACO**: Implement adaptive pheromone decay rates based on path quality.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

---

Feel free to customize any section to better describe specific aspects of your project or to provide additional usage details.



