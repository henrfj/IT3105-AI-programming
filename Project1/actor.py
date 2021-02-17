'''
Module containing the actor, in the TD actor - critic model.
'''
# Libraries.
from random import randint, uniform # Used to make epsilon random moves.
import numpy as np                  # Nice matrix operations.
import ast                          
import re

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
    
    # Selects an action based on possible actions.
    def action_selection(self, state, possible_moves, epsilon):
        '''
        Possible moves: dictionary of all childstates of state. Keys are indexes.
        Action = (state, next_state)
        If a SAP (state, possible_move) doesn't exist in the dict., 
        adds it to the dict. with value 0.
        If none of the SAPs exists in the dict., the first actions discovered will be used.
        Has a epsilon-chance of choosing randomly between possible moves anyways.
        '''
        # No possible moves
        if len(possible_moves) == 0:
            # No possible moves.
            # The state is "impossible",
            # which happens at the end of an episode.
            return -1

        # List of all keys related to possible moves.
        sub = []
        # Add all SAPS to PI
        for i in range(len(possible_moves)):
            action = possible_moves[i]
            SAP = (str(state),str(action))
            sub.append(SAP)
            try:
                a = self.PI[SAP]
            except:
                self.PI[SAP] = 0
        

        # Epsiolon greedy move
        limit = uniform(0,1)
        if (epsilon>=limit):
            # Random move
            i = randint(0,len(possible_moves)-1)
            return possible_moves[i]

        # Normalize possibilities before selection
        #self.normalize_policy(state, possible_moves)

        # Pick action to maximize output.
        '''
        best_val = 0
        best_move = possible_moves[0]
        for i in range(len(possible_moves)):
            # Needs to be str() as only immutable objects can be keys.
            action = possible_moves[i]
            SAP = (str(state),str(action))
            val = self.PI[SAP]
            if val > best_val:
                best_val = val
                best_move = action 
            
        '''
        # Max key, given subset of keys
        best_sap = max(sub, key=self.PI.get)
        # The action itself
        best_move = best_sap[1]
        # Best move is a string
        ls = re.sub('\s+', ',', best_move)
        best_move = np.array(ast.literal_eval(ls))
        
        return best_move

    # Updates the policy based on a delta.
    def update_PI(self, state, next_state, delta):
        '''
        Updates policy PI based on TD error delta.
        '''
        
        SAP = (str(state),str(next_state))
        current_policy = self.PI[SAP]
        self.PI[SAP] = current_policy + self.alpha_a * delta * self.eligibility[SAP]
    
    # Updates the eligibility, in different ways.
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
            self.eligibility[SAP] *= (self.discount * self.lambda_a)
