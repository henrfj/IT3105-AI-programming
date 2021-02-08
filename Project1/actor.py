'''
Docstring
'''


class Actor:
    '''
    Keeps the Policy and updates it througout the run.
    
    Follows an epsilon-greedy algorithm,
    to balance exploration and exploitation throughout the run.
    => Alternatively we could use boltzmanns scaling to choose stochastically.
    '''
    def __init__(self):
        # Dictionart structure(state1): [(state2_1, value), (state2_2, value), ...]
        self.PI = {}
        
    