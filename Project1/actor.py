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
        PI is an empty dictionary, later to be filled by {SAP: value} pairs.
        SAP = (str(state),str(next_state))
        '''
        # This should be filled with zeros, but we don't know all SAPs yet.
        # Therefor, the first time we see a action in self.action_selection,
        # its value will be set to zero.
        self.PI = {}

    def action_selection(self, state, possible_moves, epsilon):
        '''
        Possible moves: dictionary of all childstates of state. Keys are indexes.
        Action = (state, next_state)
        If a SAP (state, possible_move) doesn't exist in the dict., 
        adds it to the dict. with value 0.
        If none of the SAPs exists in the dict., the first possible move is returned instead.
        Has a epsilon-chance of choosing randomly between possible moves.
        '''
        if len(possible_moves) == 0:
            # No possible moves.
            # This only happends if the initial state is "impossible",
            # Or at the end of an episode.
            return -1

        limit = uniform(0,1)
        if (epsilon>=limit):
            # Random move
            i = randint(0,len(possible_moves)-1)
            return possible_moves[i]

        # Pick action to maximize
        best_val = 0
        best_move = possible_moves[0]
        
        for i in range(len(possible_moves)):
            # Needs to be str() as only immutable objects can be keys.
            action = possible_moves[i]
            SAP = (str(state),str(action))
            if (SAP in self.PI):
                val = self.PI[SAP]
                if val > best_val:
                   best_val = val
                   best_move = action 
            else: # This SAP hasn't been seen before. 
                # Adds to list of known moves. Might not be smart to do here.
                self.PI[SAP] = 0

        return best_move

    def update_PI(self, state, next_state, delta):
        pass
    
    def update_e(self, state, next_state, mode):
        '''
        Actor keeps SAP based eligibilities.
        Two modes.
        - 1: for setting it to 1, as it was just visited.
        - 2: for decaying.
        If a SAP hasn't been seen before, set its value to 0.
        '''
        SAP = (str(state),str(next_state))
        if mode==1:
            # When the state was just visited.
            self.eligibility[SAP] = 1
        if mode==2:
            # When e is decaying
            if SAP in self.eligibility:
                self.eligibility[SAP] *= (self.discount * self.lambda_a)
            else: # First time discovered
                self.eligibility[SAP] = 0



