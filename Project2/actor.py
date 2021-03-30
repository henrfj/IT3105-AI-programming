# Imports
import numpy as np

class Actor:
    '''actor class'''
    def __init__(self):
        self.NN = None


class Random_actor:
    '''Acts randomly'''
    def __init__(self, k):
        self.k = k
    
    def move_distribution(self, state):
        '''Returns a random move distribution to mimmic the real mcts actor'''
        return np.random.uniform(0,1,(self.k**2,))