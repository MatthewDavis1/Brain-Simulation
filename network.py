from enum import Enum
import random
import networkx as nx

from learning_rules import LearningRuleType, HebbianLearning
class ConnectionType(Enum):
    RANDOM = 1
    SMALL_WORLD = 2
    MODULAR = 3

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
