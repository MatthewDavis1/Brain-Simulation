import random

class Neuron:
    def __init__(self, neuron_id, leak_rate=0.95):
        self.id = neuron_id
        self.potential = 0.0
        self.firing_energy = 0.0
        self.threshold = 100.0
        self.resting_potential = 0.1
        self.refractory_period = 2
        self.refractory_counter = 0
        self.leak_factor = leak_rate
        self.fired = False

    def reset_potential(self):
        self.potential = self.resting_potential
        self.refractory_counter = self.refractory_period
        self.fired = False

    def update(self, *args, **kwargs):
        input = 0
        for pre_id in self.network.graph.predecessors(self.id):
            weight = self.network.graph[pre_id][self.id]['weight']
            pre_neuron = self.neurons[pre_id]
            input += weight * pre_neuron.firing_energy

        input += random.uniform(0, 0.1)
        self.potential += input

        if self.potential >= self.threshold:
            self.firing_energy = self.potential
            self.fired = True
            self.reset_potential()