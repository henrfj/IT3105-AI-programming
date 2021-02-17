'''
Module containing the table-based critic, of the TD actor-critic algorithm.
'''
# Libraries.
from random import uniform  # Used to initialize V(s) = some small random value.

class Critic:
    def __init__(self, discount, alpha_c, lambda_c):
        # State - based value function V(s).
        self.V = {}
        self.eligibility = {}

        # RL parameters
        self.discount = discount
        self.alpha_c = alpha_c
        self.lambda_c = lambda_c

    # Initialize V(s) for a new epoch.
    def initialize_V(self):
        '''
        Resets/initializes the value table function, for each new simulation.
        '''
        self.V = {}

    # Returns V(s).
    def evaluate(self, state):
        '''
        Returns the value of a state, given V.
        If the state is newly discovered, initialize it.
        '''
        
        key = str(state)

        try: # Assuming this key exists
            return self.V[key]
        except: # Initiate it as a small random number
            small_random_value = uniform(0,1)
            self.V[key] = small_random_value
        return self.V[key]
    
    # Updates V based on delta.
    def update_V(self, state, delta):
        '''
        Updates Value function V, based on TD-error delta.
        '''
        key = str(state)
        current_value = self.V[key]
        self.V[key] = current_value + self.alpha_c * delta * self.eligibility[key]

    # Updates e as described in the algorithm.
    def update_e(self, state, mode):
        '''
        Two modes:
        - 1: for setting it to 1, as it was just visited.
        - 2: for decaying.
        If a s hasn't been seen before, set its value to 0.
        '''
        key = str(state)
        if mode==1:
            # When the state was just visited.
            self.eligibility[key] = 1
        if mode==2:
            # When e is decaying
            self.eligibility[key] *= (self.discount * self.lambda_c)
             

