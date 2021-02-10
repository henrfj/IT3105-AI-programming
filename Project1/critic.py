'''
Docstring.
'''


class Critic:
    def __init__(self, discount, alpha_c, lambda_c):
        # Using Q(S,a): Python dictionary: key is touple of (state1, state2) - where state2 is a legal transition state. 
        # Using V(s): Key is state, returns value of state. Actor passes state to critic(*), 
        # who returns calculated TD error for given state.
        self.V = {}
        self.eligibility = {}


        # RL parameters
        self.discount = discount
        self.alpha_a = alpha_c
        self.lambda_a = lambda_c

    def initialize_V(self):
        '''
        Want to initialize with some small random value.
        Just as for the actor, as we don't know any states yet,
        these will be added in the evaluation function or smt.
        '''
        self.V = {}

    def evaluate(self, state):
        return -1

    def update_V(self, state, delta):
        pass

    def update_e(self, state, mode):
        '''
        Two modes:
        - 1: for setting it to 1, as it was just visited.
        - 2: for decaying.
        If a s hasn't been seen before, set its value to 0.
        '''
        if mode==1:
            # When the state was just visited.
            self.eligibility[state] = 1
        if mode==2:
            # When e is decaying
            if state in self.eligibility:
                self.eligibility[state] *= (self.discount * self.lambda_a)
            else: # First time discovered
                self.eligibility[state] = 0 

