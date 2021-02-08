'''
RL_engine
'''

from actor import Actor
from critic import Critic
from SW_peg_solitaire import SW

class Agent:
    '''
    Superclass of the RL engine. 
    Communicates with the simworld, passing and reading arguments.

    Usint the popular actor-ctitic version of TD(lambda) learning, with eligibility traces.
    Eligibility traces are used to update both PI and V. 
    Two modes exists:
        1. Value function stored as dictionary
        2. Value function approximated by NN

    The TD error / delta, signify the outcome of the actors move. Big delta = "better than expected", small delta.
    '''
    def __init__(self, value_mode, alpha, epochs, board_type, board_size, initial_holes):
        
        # RL parameters
        self.value_mode = value_mode
        self.learnig_rate = alpha
        self.epochs = epochs

        # The two players
        self.actor = Actor()
        self.critic = Critic()

        # The playing field
        self.sim = SW(board_type, board_size, initial_holes)
        
        # Run the simulation
        #self.simulation()

    def epsilon(self, length_episode=0, mode=0):
        '''
        Mode 0 (default): For full greed.
        
        Mode 1: fixed constant for big exploration, and thus learning.

        Mode 2: Gradual decent
        '''
        # Mode 1
        if mode == 1:
            return 0.5
        
        # Mode 2
        if mode == 2:
            if((length_episode/10000) < 1):
                return (0.5 - length_episode/10000)
            else:
                return 0
        # Mode 0
        return 0
'''
    def simulation(self):
        # Initial V with some small value
        self.critic.initialize_v()

        # Initial PI, no value associaiton: PI(s,a) = 0
        self.actor.initialize_PI()

        # Run through episodes. 
        for i in range(self.epochs):
            # Stores states in a list for animation purposes.
            episode = []

            # Create new board for the episode

            
'''




