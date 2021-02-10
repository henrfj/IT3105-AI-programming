'''
RL_engine
'''
# My own modules
from actor import Actor
from critic import Critic
from SW_peg_solitaire import SW, BT

# Libraries
import numpy as np

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
    def __init__(self, value_mode, discount, alpha_a, alpha_c, epochs, lambda_a, lambda_c, board_type, board_size, initial_holes):
        '''
        Parameters:
            -> value_mode: mode for value function. 0: dictionary, 1: NN
            ->
        '''
        # RL parameters
        self.value_mode = value_mode
        self.epochs = epochs
        self.discount = discount

        # The two players
        self.actor = Actor(discount, alpha_a, lambda_a)
        self.critic = Critic(discount, alpha_c, lambda_c)

        # The board type used for simulations field
        self.sim = SW(board_type, board_size, initial_holes)
        print("Board to learn:\n", self.sim.state)


    def learning(self, epsilon_mode):
        # Initial V with some small value
        self.critic.V = {}

        # Initial PI, no value associaiton: PI(s,a) = 0
        self.actor.PI = {}
        
        # Checks if the first state is "impossible"
        if (self.sim.final_state(self.sim.state)):
            print(self.sim.state)
            print("No learning possible, impossible state.")
            return 0

        # Run an entire epoch of episodes. i later used for plotting.
        pegs_left = [] # For plotting progression
        for i in range(self.epochs):
            # Run one episode
            print("Learning progress: "+str(100*i/self.epochs)+"%")
  
            # Stores visited states in a list.
            # Used for eligibility tracing.
            episode = []

            # Reset eligibilites
            self.actor.eligibility = {}
            self.critic.eligibility = {}

            # Reset board for this episode.
            self.sim.reset_board()

            # Initial state
            state = self.sim.state
            episode.append(state)

            # Initial action. Action = (state, next_state)
            possible_moves = self.sim.child_states(state)
            # Determine first action.
            next_state = self.actor.action_selection(state, possible_moves, self.epsilon(i, self.epochs, epsilon_mode))

            # Iterate over an episode.
            while (not self.sim.final_state(state)):
                # 1
                # Doing the action, getting a reward. Updating the board.
                reward = self.sim.reward(state, next_state)
                self.sim.set_board_state(next_state)
                

                # 2
                # Actor find next action based on the updated board-state.
                possible_moves = self.sim.child_states(next_state)
                next_move = self.actor.action_selection(next_state, possible_moves, self.epsilon(i, self.epochs, epsilon_mode))
                
                # 3
                # Eligibility updated based on "old" state s, and "old action" a
                self.actor.update_e(state, next_state, 1) # Actor keeps SAP based eligibilites

                # 4
                # Critic critisises
                delta = reward + self.discount * self.critic.evaluate(next_state) - self.critic.evaluate(state)

                # 5
                # Critic eligibility. State based eligibility. e(s) <- 1
                self.critic.update_e(state, 1)
                
                # 6
                # Do learning, based on eligibilities.
                episode.append(next_state) # Add a to the list.
                for j in range(0, len(episode)-1):
                    # Bookkeeping
                    s = episode[j]
                    a = episode[j+1]

                    # (a), (b)
                    # Critic updates value function, based on learning rate and eligibility.
                    self.critic.update_V(s, delta)
                    # Discount the e by gamma, and decay rate.
                    self.critic.update_e(s, 2) # Critic uses state-eligibilities alone.

                    # (c), (d)
                    # Actor updates policy, based on delta. (TD error)
                    self.actor.update_PI(s, a, delta)
                    # Discount the e by gamma, and decay rate. 
                    self.actor.update_e(s, a, 2)

                # 7
                # Update before proceeding
                state = np.copy(next_state)
                try: # if next-state is end-state, then next move is -1
                    next_state = np.copy(next_move)
                except:
                    next_state = next_move

                
            # How well did this run fare?
            pegs_left.append(self.sim.pegs_left(state))

        # Returned to show progression of the learning
        return pegs_left

    def run(self):
        '''

        '''
        pegs_left = []

        return pegs_left

    # Used for an epsilon-greedy algorithm.
    # TODO: plot this size
    def epsilon(self, episode_nr, epochs, mode):
        '''
        Mode 0 (default): For full greed.
        
        Mode 1: fixed constant for big exploration, and thus learning.

        Mode 2: Gradual decent
        '''
        # Mode 1
        if mode == 1:
            return 0.3
        
        # Mode 2
        if mode == 2:
            return (0.5 - 0.5*episode_nr/epochs)
            
        # Mode 0
        return 0

