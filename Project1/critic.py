'''
Docstring.
'''


class Critic:
    def __init__(self, discount, alpha_c, lambda_c):
        # Using Q(S,a): Python dictionary: key is touple of (state1, state2) - where state2 is a legal transition state. 
        # Using V(s): Key is state, returns value of state. Actor passes state to critic(*), 
        # who returns calculated TD error for given state.
        self.V = {}
        self.eligibiliy = {}


        # RL parameters
        self.discount = discount
        self.alpha_a = alpha_c
        self.lambda_a = lambda_c

    def initialize_V(self):
        pass

    def evaluate(self, state):
        return -1

    def update_V(self, state, delta):
        pass

    def update_e(self, state):
        pass 

