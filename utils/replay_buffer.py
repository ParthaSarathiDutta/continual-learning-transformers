from collections import deque
import random

class ReplayBuffer:
    def __init__(self, max_size=100):
        self.buffer = deque(maxlen=max_size)

    def add_batch(self, dataset):
        for item in dataset:
            self.buffer.append(item)

    def sample(self, size):
        return random.sample(self.buffer, min(len(self.buffer), size))
