from enum import Enum

class LearningRuleType(Enum):
    HEBBIAN = 1

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