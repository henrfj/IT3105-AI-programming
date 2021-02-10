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
    # TODO: look into normalizing the policy distribution. 
    def __init__(self, discount, alpha_a, lambda_a):
        # Dictionart structure(state1): [(state2_1, value), (state2_2, value), ...]
        self.PI = {}
        self.eligibility = {} #  SAP pairs (state, next_state)

        # RL parameters
        self.discount = discount
        self.alpha_a = alpha_a
        self.lambda_a = lambda_a
    
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

        # Add all SAPS to PI
        for i in range(len(possible_moves)):
            action = possible_moves[i]
            SAP = (str(state),str(action))
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
            

        return best_move

    def update_PI(self, state, next_state, delta):
        '''
        Updates policy PI based on TD error delta.
        '''
        
        SAP = (str(state),str(next_state))
        current_policy = self.PI[SAP]
        self.PI[SAP] = current_policy + self.alpha_a * delta * self.eligibility[SAP]
        
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
            
    def normalize_policy(self, state, possible_moves):
        '''
        Normalizing SAPs from a single state. Makes it into a distribution,
        that could be used for selection (but we use epsilon). 
        '''
        # Find scale, to normalize negative and positive numbers
        minimum = 0
        for i in range(len(possible_moves)):
            action = possible_moves[i]
            SAP = (str(state),str(action))
            if self.PI[SAP] < minimum:
                minimum = self.PI[SAP]

        # Normalize
        total = 0
        for i in range(len(possible_moves)):
            action = possible_moves[i]
            SAP = (str(state),str(action)) 
            self.PI[SAP] += (abs(minimum)) #+ 1)
            total += self.PI[SAP]
            

        for i in range(len(possible_moves)):
            action = possible_moves[i]
            SAP = (str(state),str(action))
            try: # AS long as total =/= 0
                self.PI[SAP] *= 1/total
            except:
                pass
        
        # For debugging
        actions = []
        for i in range(len(possible_moves)):
            action = possible_moves[i]
            SAP = (str(state),str(action))
            actions.append(self.PI[SAP])
        print("Normalized selection:" + str(actions))

