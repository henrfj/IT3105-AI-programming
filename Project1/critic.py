'''
Docstring.
'''
from random import uniform

class Critic:
    def __init__(self, discount, alpha_c, lambda_c):
        # Using Q(S,a): Python dictionary: key is touple of (state1, state2) - where state2 is a legal transition state. 
        # Using V(s): Key is state, returns value of state. Actor passes state to critic(*), 
        # who returns calculated TD error for given state.
        self.V = {}
        self.eligibility = {}


        # RL parameters
        self.discount = discount
        self.alpha_c = alpha_c
        self.lambda_c = lambda_c

    def evaluate(self, state):
        key = str(state)

        try: # Assuming this key exists
            return self.V[key]
        except: # Initiate it as a small random number
            small_random_value = uniform(0,1)
            self.V[key] = small_random_value
        return self.V[key]
        
    def update_V(self, state, delta):
        '''
        Updates Value function V, based on TD-error delta.
        '''
        key = str(state)
        current_value = self.V[key]
        self.V[key] = current_value + self.alpha_c * delta * self.eligibility[key]

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
             

