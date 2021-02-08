'''
Docstring.
'''


class Critic:
    def __init__(self):
        # Using Q(S,a): Python dictionary: key is touple of (state1, state2) - where state2 is a legal transition state. 
        # Using V(s): Key is state, returns value of state. Actor passes state to critic(*), 
        # who returns calculated TD error for given state.
        #self.Q = {}
        self.V = {}