'''
Docstring
'''
# used for random-epsilon action selection.
from random import randint, uniform

class Actor:
    '''
    Keeps the Policy and updates it througout the run.
    
    Follows an epsilon-greedy algorithm,
    to balance exploration and exploitation throughout the run.
    => Alternatively we could use boltzmanns scaling to choose stochastically.
    '''
    def __init__(self, discount, alpha_a, lambda_a):
        # Dictionart structure(state1): [(state2_1, value), (state2_2, value), ...]
        self.PI = {}
        self.eligibility = {} #  SAP pairs (state, next_state)

        # RL parameters
        self.discount = discount
        self.alpha_a = alpha_a
        self.lambda_a = lambda_a
    
    def initialize_PI(self):
        '''
        PI is an empty dictionary
        '''
        self.PI = {}

    def action_selection(self, state, possible_moves, epsilon):
        '''
        Possible moves: dictionary of all childstates of state. Keys are indexes.
        Action = (state, next_state)
        '''
        limit = uniform(0,1)
        if (epsilon>=limit):
            # Random move
            i = randint(0,len(possible_moves)-1)
            return possible_moves[i]

        # Pick action to maximize
        best_val = 0
        best_move = possible_moves[0]
        
        for action in possible_moves:
            # Needs to be str() as only immutable objects can be keys.
            SAP = (str(state),str(action))
            if (SAP in self.PI):
                val = self.PI[SAP]
                if val > best_val:
                   best_val = val
                   best_move = action 
            else: # Adds it to the policy. Might not be smart to do here.
                self.PI[SAP] = 0

        return best_move

    def update_PI(self, state, next_state, delta):
        pass
    
    def update_e(self, state, next_state):
        '''
        Actor keeps SAP based eligibilities.
        Update the eligibility.
        "Action" is just next state.
        '''
        pass



