#!/usr/bin/env python3

import random
import networkx as nx
from enum import Enum
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.animation import FuncAnimation

class ConnectionType(Enum):
    RANDOM = 1
    SMALL_WORLD = 2

class Neuron:
    def __init__(self, neuron_id):
        self.id = neuron_id
        self.potential = 0.0
        self.firing_energy = 0.0
        self.threshold = 1.0  # Firing threshold
        self.resting_potential = -70.0  # Changed from 0.0 to match min_potential
        self.refractory_period = 2  # Time steps
        self.refractory_counter = 0
        self.min_potential = -70.0  # resting potential in mV
        self.max_potential = 40.0   # peak potential in mV
        self.leak_factor = 0.95    # Added: prevents immediate decay to 0
        self.fired = False         # Added: track firing state

    def reset_potential(self):
        self.potential = self.resting_potential
        self.refractory_counter = self.refractory_period
        self.fired = False

    def update(self, *args, **kwargs):
        # Get input from connected neurons
        input_current = 0
        for pre_id in self.network.graph.predecessors(self.id):
            weight = self.network.graph[pre_id][self.id]['weight']
            pre_neuron = self.neurons[pre_id]
            input_current += weight * pre_neuron.firing_energy

        # Add some random noise
        input_current += random.uniform(0, 0.1)
        
        self.potential += input_current
        
        # Clamp the potential to biological range
        self.potential = max(self.min_potential, min(self.max_potential, self.potential))
        
        # Check for firing
        if self.potential >= self.threshold:
            self.firing_energy = self.potential
            self.min_firing_energy = min(self.min_firing_energy, self.firing_energy)
            self.max_firing_energy = max(self.max_firing_energy, self.firing_energy)
            self.global_energy_levels.append(self.firing_energy)
            self.fired = True
            self.reset_potential()

class LearningRuleType(Enum):
    HEBBIAN = 1
    # Future learning rules can be added here
    # e.g., STDP = 2

class LearningRule:
    def apply(self, network, pre_neuron, post_neuron):
        raise NotImplementedError("This method should be overridden by subclasses.")

class HebbianLearning(LearningRule):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def apply(self, network, pre_neuron, post_neuron):
        # Hebbian update: Δw = learning_rate * pre_activity * post_activity
        pre_activity = pre_neuron.firing_energy if pre_neuron.firing_energy > 0 else 0
        post_activity = post_neuron.firing_energy if post_neuron.firing_energy > 0 else 0
        delta_w = self.learning_rate * pre_activity * post_activity

        # Update the weight, ensuring it stays within [0, 1]
        current_weight = network.graph[pre_neuron.id][post_neuron.id]['weight']
        new_weight = min(max(current_weight + delta_w, 0.0), 1.0)
        network.graph[pre_neuron.id][post_neuron.id]['weight'] = new_weight

class Network:
    def __init__(self, N, connection_type, learning_rule_type=LearningRuleType.HEBBIAN):
        self.N = N
        self.connection_type = connection_type
        # Parameters for small-world network
        self.local_density = 0.8  # High local clustering
        self.long_range_prob = 0.1  # Sparse long-range connections
        self.distance_decay = 2.0  # Power law decay for connection probability
        self.graph = self.generate_graph()
        self.learning_rule = self.initialize_learning_rule(learning_rule_type)

    def calculate_connection_probability(self, distance):
        """Calculate connection probability based on distance using a power law."""
        return self.local_density * (1.0 / (1.0 + (distance ** self.distance_decay)))

    def generate_graph(self):
        if self.connection_type == ConnectionType.RANDOM:
            G = nx.erdos_renyi_graph(n=self.N, p=0.5, directed=True)
            for (u, v) in G.edges():
                G[u][v]['weight'] = random.uniform(0.5, 1.0)
        elif self.connection_type == ConnectionType.SMALL_WORLD:
            G = nx.DiGraph()
            G.add_nodes_from(range(self.N))
            
            # Create initial local connections with high clustering
            for i in range(self.N):
                # Local connections within neighborhood
                neighborhood_size = int(self.N * 0.1)  # 10% of neurons are local neighbors
                for j in range(max(0, i-neighborhood_size), min(self.N, i+neighborhood_size)):
                    if i != j:
                        distance = min(abs(i-j), self.N - abs(i-j))  # Consider circular boundary
                        prob = self.calculate_connection_probability(distance)
                        if random.random() < prob:
                            G.add_edge(i, j, weight=random.uniform(0.5, 1.0))  # Initialize with random weight
                    
                # Add sparse long-range connections
                for j in range(self.N):
                    if i != j and not G.has_edge(i, j):
                        if random.random() < self.long_range_prob:
                            G.add_edge(i, j, weight=random.uniform(0.5, 1.0))  # Initialize with random weight
            
            # Verify and ensure the graph is connected
            if not nx.is_strongly_connected(G):
                components = list(nx.strongly_connected_components(G))
                # Connect components with minimal additional edges
                for idx in range(len(components)-1):
                    node1 = random.choice(list(components[idx]))
                    node2 = random.choice(list(components[idx+1]))
                    G.add_edge(node1, node2, weight=random.uniform(0.5, 1.0))
        else:
            raise ValueError("Unsupported connection type")
        
        # Calculate and store network metrics
        self.clustering_coefficient = nx.average_clustering(G, weight='weight')
        self.average_path_length = nx.average_shortest_path_length(G, weight=None)  # Unweighted path length
        
        return G

    def initialize_learning_rule(self, learning_rule_type):
        if learning_rule_type == LearningRuleType.HEBBIAN:
            return HebbianLearning()
        # Future learning rules can be initialized here
        else:
            raise ValueError("Unsupported learning rule type")

    def get_network_metrics(self):
        """Return key metrics about the network structure."""
        return {
            'clustering_coefficient': self.clustering_coefficient,
            'average_path_length': self.average_path_length,
            'density': nx.density(self.graph),
            'degree_assortativity': nx.degree_assortativity_coefficient(self.graph),
            'average_degree': sum(dict(self.graph.degree()).values()) / self.N
        }

class Simulation:
    def __init__(self, N, connection_type, learning_rule_type=LearningRuleType.HEBBIAN, 
                 init_potential_mean=0.5, init_potential_std=0.2):
        self.network = Network(N, connection_type, learning_rule_type)
        self.neurons = [Neuron(i) for i in range(N)]
        
        # Initialize all neurons with gaussian-distributed potentials
        for neuron in self.neurons:
            # Clip potential between 0 and threshold to ensure valid values
            neuron.potential = np.clip(
                np.random.normal(init_potential_mean, init_potential_std),
                0.0,
                neuron.threshold
            )
        
        self.time_steps = 0
        self.min_firing_energy = float('inf')
        self.max_firing_energy = float('-inf')
        self.global_energy_levels = []
        self.firing_events_history = []
        self.potential_history = []
        self.hubs = self.identify_hubs()
        self.network_metrics = self.network.get_network_metrics()
        
        # Record initial state
        self.record_potentials()

    def record_potentials(self):
        """Record the potential of each neuron at current timestep"""
        self.potential_history.append([n.potential for n in self.neurons])

    def identify_hubs(self):
        degree_dict = dict(self.network.graph.degree())
        max_degree = max(degree_dict.values())
        hubs = [node for node, degree in degree_dict.items() if degree == max_degree]
        return hubs

    def advance_time_step(self):
        fired_neurons = []
        for neuron in self.neurons:
            if neuron.refractory_counter > 0:
                neuron.refractory_counter -= 1
                continue

            # Apply leak current (gradual decay towards resting potential)
            potential_diff = neuron.potential - neuron.resting_potential
            neuron.potential = neuron.resting_potential + (potential_diff * neuron.leak_factor)

            # Get input from connected neurons
            input_current = 0
            for pre_id in self.network.graph.predecessors(neuron.id):
                weight = self.network.graph[pre_id][neuron.id]['weight']
                pre_neuron = self.neurons[pre_id]
                if pre_neuron.fired:  # Only consider input from neurons that fired
                    input_current += weight * 30.0  # Fixed amplitude for firing input

            # Add some random noise (scaled appropriately)
            input_current += random.uniform(-5, 5)  # Biological scale noise
            
            # Update potential
            neuron.potential += input_current
            
            # Clamp potential to biological range
            neuron.potential = max(neuron.min_potential, 
                                 min(neuron.max_potential, neuron.potential))
            
            # Check for firing
            if neuron.potential >= neuron.threshold:
                neuron.firing_energy = neuron.potential - neuron.resting_potential
                self.min_firing_energy = min(self.min_firing_energy, neuron.firing_energy)
                self.max_firing_energy = max(self.max_firing_energy, neuron.firing_energy)
                self.global_energy_levels.append(neuron.firing_energy)
                fired_neurons.append(neuron)
                neuron.fired = True
                neuron.reset_potential()
            else:
                neuron.fired = False

        self.firing_events_history.append(len(fired_neurons))
        self.time_steps += 1

        # Apply learning rule based on fired neurons
        self.apply_learning_rule(fired_neurons)
        
        # Record potentials
        self.record_potentials()

    def apply_learning_rule(self, fired_neurons):
        for neuron in fired_neurons:
            # Iterate over all post-synaptic neurons connected to the fired neuron
            for post_neuron_id in self.network.graph.neighbors(neuron.id):
                post_neuron = self.neurons[post_neuron_id]
                self.network.learning_rule.apply(self.network, neuron, post_neuron)

    def print_neuron_state(self, neuron_id):
        if 0 <= neuron_id < len(self.neurons):
            neuron = self.neurons[neuron_id]
            print(f"Neuron {neuron.id}: Potential = {neuron.potential:.2f}")
        else:
            print("Invalid neuron ID.")

    def print_all_neurons(self):
        for neuron in self.neurons:
            print(f"Neuron {neuron.id}: Potential = {neuron.potential:.2f}")

    def run_simulation(self, total_steps):
        for _ in range(total_steps):
            self.advance_time_step()
        stats = {
            "time_steps": self.time_steps,
            "min_firing_energy": self.min_firing_energy,
            "max_firing_energy": self.max_firing_energy,
            "num_global_energy_levels": len(self.global_energy_levels),
            "num_distinct_hubs": len(self.hubs),
            "network_metrics": self.network_metrics
        }
        return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Network Simulation with LIF Neurons')
    parser.add_argument('-n', '--neurons', type=int, default=100,
                      help='Number of neurons in the network (default: 100)')
    parser.add_argument('-t', '--timesteps', type=int, default=1000,
                      help='Number of simulation timesteps (default: 1000)')
    parser.add_argument('-c', '--connection', type=str, choices=['random', 'small_world'],
                      default='random', help='Type of network connection (default: random)')
    parser.add_argument('-l', '--learning', type=str, choices=['hebbian'],
                      default='hebbian', help='Type of learning rule (default: hebbian)')
    parser.add_argument('--no-plot', action='store_true',
                      help='Disable network visualization')
    parser.add_argument('-o', '--output', type=str,
                      help='Output file for statistics (optional)')
    parser.add_argument('--init-mean', type=float, default=0.5,
                      help='Mean of initial potential distribution (default: 0.5)')
    parser.add_argument('--init-std', type=float, default=0.2,
                      help='Standard deviation of initial potential distribution (default: 0.2)')
    
    args = parser.parse_args()
    
    # Convert connection type string to enum
    connection_map = {
        'random': ConnectionType.RANDOM,
        'small_world': ConnectionType.SMALL_WORLD
    }
    connection_type = connection_map[args.connection]
    
    # Convert learning rule string to enum
    learning_map = {
        'hebbian': LearningRuleType.HEBBIAN
        # Future learning rules can be added here
    }
    learning_rule_type = learning_map[args.learning]
    
    # Initialize and run simulation with new parameters
    sim = Simulation(
        args.neurons, 
        connection_type, 
        learning_rule_type,
        init_potential_mean=args.init_mean,
        init_potential_std=args.init_std
    )
    stats = sim.run_simulation(total_steps=args.timesteps)
    
    # Print statistics
    stats_output = [
        "Simulation Statistics:",
        f"Total Time-Steps: {stats['time_steps']}",
        f"Min Firing Energy: {stats['min_firing_energy']:.4f}",
        f"Max Firing Energy: {stats['max_firing_energy']:.4f}",
        f"Number of Global Energy Levels: {stats['num_global_energy_levels']}",
        f"Number of Distinct Hubs: {stats['num_distinct_hubs']}",
        "\nNetwork Metrics:",
        f"Clustering Coefficient: {stats['network_metrics']['clustering_coefficient']:.4f}",
        f"Average Path Length: {stats['network_metrics']['average_path_length']:.4f}",
        f"Network Density: {stats['network_metrics']['density']:.4f}",
        f"Degree Assortativity: {stats['network_metrics']['degree_assortativity']:.4f}",
        f"Average Degree: {stats['network_metrics']['average_degree']:.4f}"
    ]
    
    # Print to stdout
    print('\n'.join(stats_output))
    
    # Write to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            f.write('\n'.join(stats_output))
    
    # Only create the animation
    def create_potential_animation(sim):
        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(sim.network.graph)
        
        # Get edge weights for scaling
        weights = [sim.network.graph[u][v]['weight'] for u, v in sim.network.graph.edges()]
        max_weight = max(weights)
        min_weight = min(weights)
        
        # Add this before the animation creation
        # Save neuron potentials to file
        with open('neuron_potentials.csv', 'w') as f:
            # Write header
            f.write('Timestep,' + ','.join([f'Neuron_{i}' for i in range(len(sim.network.graph.nodes()))]) + '\n')
            
            # Write data for each timestep
            for t, potentials in enumerate(sim.potential_history):
                f.write(f'{t},' + ','.join([f'{p:.4f}' for p in potentials]) + '\n')
        
        def update(frame):
            ax.clear()
            potentials = sim.potential_history[frame]
            
            # Create color map (blue for negative, white for 0, red for positive)
            colors = []
            for potential in potentials:
                if potential <= 0:
                    # Blue to white gradient
                    intensity = min(1.0, abs(potential) / abs(sim.neurons[0].min_potential))  # Normalize against min_potential
                    colors.append((1-intensity, 1-intensity, 1))
                else:
                    # White to red gradient
                    intensity = min(1.0, potential / sim.neurons[0].max_potential)  # Normalize against max_potential
                    colors.append((1, 1-intensity, 1-intensity))
            
            # Draw edges with width based on weight
            edge_widths = []
            for (u, v) in sim.network.graph.edges():
                weight = sim.network.graph[u][v]['weight']
                # Avoid division by zero by checking if all weights are the same
                if max_weight == min_weight:
                    width = 2.5  # Use a default middle value
                else:
                    width = 0.5 + 4.5 * (weight - min_weight) / (max_weight - min_weight)
                edge_widths.append(width)
            
            nx.draw(sim.network.graph, pos,
                   node_color=colors,
                   node_size=500,
                   with_labels=True,
                   edge_color='gray',
                   width=edge_widths,  # Add varying edge widths
                   alpha=0.6,
                   ax=ax)
            
            ax.set_title(f'Time Step: {frame}')
        
        anim = FuncAnimation(
            fig, 
            update,
            frames=len(sim.potential_history),
            interval=200,  # 200ms between frames
            repeat=True
        )
        
        # Save animation
        anim.save('neural_activity.gif', writer='pillow')
        plt.close()
    
    create_potential_animation(sim)