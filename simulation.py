#!/usr/bin/env python3

import os
import random
import argparse
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

class ConnectionType(Enum):
    RANDOM = 1
    SMALL_WORLD = 2
    MODULAR = 3

class LearningRuleType(Enum):
    HEBBIAN = 1

class Neuron:
    def __init__(self, neuron_id, leak_rate=0.95):
        self.id = neuron_id
        self.potential = 0.0
        self.firing_energy = 0.0
        self.threshold = 1.0
        self.resting_potential = -70.0
        self.refractory_period = 2
        self.refractory_counter = 0
        self.min_potential = -70.0
        self.max_potential = 40.0
        self.leak_factor = leak_rate
        self.fired = False

    def reset_potential(self):
        self.potential = self.resting_potential
        self.refractory_counter = self.refractory_period
        self.fired = False

    def update(self, *args, **kwargs):
        input_current = 0
        for pre_id in self.network.graph.predecessors(self.id):
            weight = self.network.graph[pre_id][self.id]['weight']
            pre_neuron = self.neurons[pre_id]
            input_current += weight * pre_neuron.firing_energy

        input_current += random.uniform(0, 0.1)
        self.potential += input_current
        self.potential = max(self.min_potential, min(self.max_potential, self.potential))
        
        if self.potential >= self.threshold:
            self.firing_energy = self.potential
            self.min_firing_energy = min(self.min_firing_energy, self.firing_energy)
            self.max_firing_energy = max(self.max_firing_energy, self.firing_energy)
            self.global_energy_levels.append(self.firing_energy)
            self.fired = True
            self.reset_potential()

class LearningRule:
    def apply(self, network, pre_neuron, post_neuron):
        raise NotImplementedError("This method should be overridden by subclasses.")

class HebbianLearning(LearningRule):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def apply(self, network, pre_neuron, post_neuron):
        pre_activity = pre_neuron.firing_energy if pre_neuron.firing_energy > 0 else 0
        post_activity = post_neuron.firing_energy if post_neuron.firing_energy > 0 else 0
        delta_w = self.learning_rate * pre_activity * post_activity
        current_weight = network.graph[pre_neuron.id][post_neuron.id]['weight']
        new_weight = min(max(current_weight + delta_w, 0.0), 1.0)
        network.graph[pre_neuron.id][post_neuron.id]['weight'] = new_weight

class Network:
    def __init__(self, N, connection_type, learning_rule_type=LearningRuleType.HEBBIAN):
        self.N = N
        self.connection_type = connection_type
        self.local_density = 0.8
        self.long_range_prob = 0.1
        self.distance_decay = 2.0
        self.graph = self.generate_graph()
        self.learning_rule = self.initialize_learning_rule(learning_rule_type)

    def calculate_connection_probability(self, distance):
        return self.local_density * (1.0 / (1.0 + (distance ** self.distance_decay)))

    def generate_graph(self):
        if self.connection_type == ConnectionType.RANDOM:
            G = nx.erdos_renyi_graph(n=self.N, p=0.5, directed=True)
            for (u, v) in G.edges():
                G[u][v]['weight'] = random.uniform(0.5, 1.0)
        elif self.connection_type == ConnectionType.SMALL_WORLD:
            G = nx.DiGraph()
            G.add_nodes_from(range(self.N))
            
            for i in range(self.N):
                neighborhood_size = int(self.N * 0.1)
                for j in range(max(0, i-neighborhood_size), min(self.N, i+neighborhood_size)):
                    if i != j:
                        distance = min(abs(i-j), self.N - abs(i-j))
                        prob = self.calculate_connection_probability(distance)
                        if random.random() < prob:
                            G.add_edge(i, j, weight=random.uniform(0.5, 1.0))
                    
                for j in range(self.N):
                    if i != j and not G.has_edge(i, j):
                        if random.random() < self.long_range_prob:
                            G.add_edge(i, j, weight=random.uniform(0.5, 1.0))
            
            if not nx.is_strongly_connected(G):
                components = list(nx.strongly_connected_components(G))
                for idx in range(len(components)-1):
                    node1 = random.choice(list(components[idx]))
                    node2 = random.choice(list(components[idx+1]))
                    G.add_edge(node1, node2, weight=random.uniform(0.5, 1.0))
        elif self.connection_type == ConnectionType.MODULAR:
            G = nx.DiGraph()
            G.add_nodes_from(range(self.N))
            
            num_modules = min(4, self.N // 10)
            neurons_per_module = self.N // num_modules
            within_module_p = 0.7
            
            for module in range(num_modules):
                start_idx = module * neurons_per_module
                end_idx = start_idx + neurons_per_module if module < num_modules - 1 else self.N
                
                module_neurons = list(range(start_idx, end_idx))
                for i in module_neurons:
                    for j in module_neurons:
                        if i != j and random.random() < within_module_p:
                            G.add_edge(i, j, weight=random.uniform(0.5, 1.0))
                
                if module < num_modules - 1:
                    source = random.choice(module_neurons)
                    target_module = module + 1
                    target_start = target_module * neurons_per_module
                    target_end = target_start + neurons_per_module
                    target = random.randint(target_start, target_end-1)
                    G.add_edge(source, target, weight=random.uniform(0.5, 1.0))
            
            if not nx.is_strongly_connected(G):
                components = list(nx.strongly_connected_components(G))
                for idx in range(len(components)-1):
                    node1 = random.choice(list(components[idx]))
                    node2 = random.choice(list(components[idx+1]))
                    G.add_edge(node1, node2, weight=random.uniform(0.5, 1.0))
        else:
            raise ValueError("Unsupported connection type")
        
        self.clustering_coefficient = nx.average_clustering(G, weight='weight')
        self.average_path_length = nx.average_shortest_path_length(G, weight=None)
        
        return G

    def initialize_learning_rule(self, learning_rule_type):
        if learning_rule_type == LearningRuleType.HEBBIAN:
            return HebbianLearning()
        raise ValueError("Unsupported learning rule type")

    def get_network_metrics(self):
        return {
            'clustering_coefficient': self.clustering_coefficient,
            'average_path_length': self.average_path_length,
            'density': nx.density(self.graph),
            'degree_assortativity': nx.degree_assortativity_coefficient(self.graph),
            'average_degree': sum(dict(self.graph.degree()).values()) / self.N
        }

class Simulation:
    def __init__(self, N, connection_type, learning_rule_type=LearningRuleType.HEBBIAN, 
                 init_potential_mean=0.5, init_potential_std=0.2, output_dir='output', 
                 leak_rate=0.95, num_drivers=0, driver_potential=1.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.network = Network(N, connection_type, learning_rule_type)
        self.neurons = [Neuron(i, leak_rate) for i in range(N)]
        
        for neuron in self.neurons:
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
        
        self.num_drivers = num_drivers
        self.driver_potential = driver_potential
        # Randomly select neurons to be drivers
        self.driver_neurons = random.sample(range(N), num_drivers) if num_drivers > 0 else []
        
        self.record_potentials()

    def record_potentials(self):
        self.potential_history.append([n.potential for n in self.neurons])

    def identify_hubs(self):
        degree_dict = dict(self.network.graph.degree())
        max_degree = max(degree_dict.values())
        hubs = [node for node, degree in degree_dict.items() if degree == max_degree]
        return hubs

    def advance_time_step(self):
        # Phase 1: Compute the new potentials and firing status without updating the neurons
        fired_previous = [neuron.fired for neuron in self.neurons]
        new_potentials = [neuron.potential for neuron in self.neurons]
        new_fired_neurons = [False for _ in self.neurons]
        new_refractory_counters = [neuron.refractory_counter for neuron in self.neurons]
        fired_neurons = []

        for idx, neuron in enumerate(self.neurons):
            # If this is a driver neuron, add constant input
            if idx in self.driver_neurons:
                new_potentials[idx] += self.driver_potential
                continue

            if fired_previous[idx]:
                # Neuron was firing in the previous step; handle refractory period
                if neuron.refractory_counter > 0:
                    new_refractory_counters[idx] -= 1
                continue

            if neuron.refractory_counter > 0:
                new_refractory_counters[idx] -= 1
                continue

            potential_diff = neuron.potential - neuron.resting_potential
            updated_potential = neuron.resting_potential + (potential_diff * neuron.leak_factor)

            input_current = 0
            for pre_id in self.network.graph.predecessors(neuron.id):
                weight = self.network.graph[pre_id][neuron.id]['weight']
                if fired_previous[pre_id]:
                    pre_neuron = self.neurons[pre_id]
                    input_current += weight * pre_neuron.firing_energy

            input_current += random.uniform(-5, 5)
            updated_potential += input_current
            updated_potential = max(neuron.min_potential, min(neuron.max_potential, updated_potential))

            new_potentials[idx] = updated_potential

            if updated_potential >= neuron.threshold:
                firing_energy = updated_potential - neuron.resting_potential
                self.min_firing_energy = min(self.min_firing_energy, firing_energy)
                self.max_firing_energy = max(self.max_firing_energy, firing_energy)
                self.global_energy_levels.append(firing_energy)
                new_fired_neurons[idx] = True
                fired_neurons.append(neuron)
                new_potentials[idx] = neuron.resting_potential
                new_refractory_counters[idx] = neuron.refractory_period

        # Phase 2: Update all neurons with the computed potentials and firing status
        for idx, neuron in enumerate(self.neurons):
            neuron.potential = new_potentials[idx]
            neuron.fired = new_fired_neurons[idx]
            neuron.refractory_counter = new_refractory_counters[idx]

        self.firing_events_history.append(len(fired_neurons))
        self.time_steps += 1

        self.apply_learning_rule(fired_neurons)
        self.record_potentials()

    def apply_learning_rule(self, fired_neurons):
        for neuron in fired_neurons:
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
        self.save_potential_history()
        return stats

    def save_potential_history(self):
        """Save the neuron potential history to a CSV file."""
        df = pd.DataFrame(self.potential_history)
        df.insert(0, 'Timestep', range(len(self.potential_history)))
        output_file = self.output_dir / 'neuron_potentials.csv'
        df.to_csv(output_file, index=False)

def main():
    parser = argparse.ArgumentParser(description='Neural Network Simulation with LIF Neurons')
    parser.add_argument('-n', '--neurons', type=int, default=100,
                      help='Number of neurons in the network (default: 100)')
    parser.add_argument('-t', '--timesteps', type=int, default=1000,
                      help='Number of simulation timesteps (default: 1000)')
    parser.add_argument('-c', '--connection', type=str, choices=['random', 'small_world', 'modular'],
                      default='random', help='Type of network connection (default: random)')
    parser.add_argument('-l', '--learning', type=str, choices=['hebbian'],
                      default='hebbian', help='Type of learning rule (default: hebbian)')
    parser.add_argument('--no-plot', action='store_true',
                      help='Disable network visualization')
    parser.add_argument('-o', '--output-dir', type=str, default='output',
                      help='Directory for output files (default: output)')
    parser.add_argument('--init-mean', type=float, default=0.5,
                      help='Mean of initial potential distribution (default: 0.5)')
    parser.add_argument('--init-std', type=float, default=0.2,
                      help='Standard deviation of initial potential distribution (default: 0.2)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                      help='Learning rate for synaptic weight updates (default: 0.01)')
    parser.add_argument('--stats-file', type=str,
                      help='Output file name for statistics (optional)')
    parser.add_argument('--leak-rate', type=float, default=0.95,
                        help='Leak rate for neuron potential decay (default: 0.95)')
    parser.add_argument('--num-drivers', type=int, default=0,
                      help='Number of driving input neurons (default: 0)')
    parser.add_argument('--driver-potential', type=float, default=1.0,
                      help='Constant potential added to driver neurons each timestep (default: 1.0)')
    
    args = parser.parse_args()
    
    connection_type = {
        'random': ConnectionType.RANDOM,
        'small_world': ConnectionType.SMALL_WORLD,
        'modular': ConnectionType.MODULAR
    }[args.connection]
    
    learning_rule_type = {
        'hebbian': LearningRuleType.HEBBIAN
    }[args.learning]
    
    sim = Simulation(
        args.neurons, 
        connection_type, 
        learning_rule_type,
        init_potential_mean=args.init_mean,
        init_potential_std=args.init_std,
        output_dir=args.output_dir,
        leak_rate=args.leak_rate,
        num_drivers=args.num_drivers,
        driver_potential=args.driver_potential
    )
    
    sim.network.learning_rule.learning_rate = args.learning_rate
    stats = sim.run_simulation(total_steps=args.timesteps)
    
    stats_output = [
        "Simulation Statistics:",
        f"Total Time-Steps: {stats['time_steps']}",
        f"Min Firing Energy: {stats['min_firing_energy']:.4f}",
        f"Max Firing Energy: {stats['max_firing_energy']:.4f}",
        f"Number of Global Energy Levels: {stats['num_global_energy_levels']}",
        f"Number of Distinct Hubs: {stats['num_distinct_hubs']}",
        f"Learning Rate: {args.learning_rate}",
        "\nNetwork Metrics:",
        f"Clustering Coefficient: {stats['network_metrics']['clustering_coefficient']:.4f}",
        f"Average Path Length: {stats['network_metrics']['average_path_length']:.4f}",
        f"Network Density: {stats['network_metrics']['density']:.4f}",
        f"Degree Assortativity: {stats['network_metrics']['degree_assortativity']:.4f}",
        f"Average Degree: {stats['network_metrics']['average_degree']:.4f}"
    ]
    
    print('\n'.join(stats_output))
    
    if args.stats_file:
        stats_file = os.path.join(args.output_dir, args.stats_file)
        with open(stats_file, 'w') as f:
            f.write('\n'.join(stats_output))
    
    if not args.no_plot:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        pos = nx.spring_layout(sim.network.graph, seed=42)
        
        edges = list(sim.network.graph.edges())
        edge_weights = [sim.network.graph[u][v]['weight'] for u, v in edges]
        max_weight = max(edge_weights) if edge_weights else 1
        min_weight = min(edge_weights) if edge_weights else 0.5
        normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in edge_weights]
        edge_colors = ['gray'] * len(edges)
        edge_transparency = 0.6

        edge_lines = [(pos[u], pos[v]) for u, v in edges]
        lc = LineCollection(edge_lines, colors=edge_colors, linewidths=[0.5 + 4.5 * nw for nw in normalized_weights],
                           alpha=edge_transparency)
        ax.add_collection(lc)

        initial_potentials = sim.potential_history[0]
        nodes = sim.network.graph.nodes()
        scatter = ax.scatter(
            [pos[node][0] for node in nodes],
            [pos[node][1] for node in nodes],
            c=initial_potentials,
            cmap='bwr',
            vmin=sim.neurons[0].min_potential,
            vmax=sim.neurons[0].max_potential,
            s=500
        )

        ax.set_xlim(min(x for x, y in pos.values()) - 0.1, max(x for x, y in pos.values()) + 0.1)
        ax.set_ylim(min(y for x, y in pos.values()) - 0.1, max(y for x, y in pos.values()) + 0.1)
        ax.set_title('Time Step: 0')
        ax.axis('off')

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Neuron Potential (mV)')

        def update(frame):
            if frame >= len(sim.potential_history):
                return scatter,
            
            potentials = sim.potential_history[frame]
            scatter.set_array(np.array(potentials))
            ax.set_title(f'Time Step: {frame}')
            return scatter,

        anim = FuncAnimation(
            fig, 
            update,
            frames=len(sim.potential_history),
            interval=50,
            blit=True
        )

        try:
            Writer = matplotlib.animation.FFMpegWriter
            writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
            animation_file = os.path.join(sim.output_dir, 'neural_activity.mp4')
            anim.save(animation_file, writer=writer)
        except Exception as e:
            print("FFMpegWriter not available, falling back to pillow. Install ffmpeg for better performance.")
            animation_file = os.path.join(sim.output_dir, 'neural_activity.gif')
            anim.save(animation_file, writer='pillow')
        
        print(f"Animation saved to {animation_file}")
        plt.close()

if __name__ == "__main__":
    main()