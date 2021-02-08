'''
RL_engine
'''

from actor import Actor
from critic import Critic
from SW_peg_solitaire import SW, BT

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
        
        # Run the learning simulation
        self.learning()
        # TODO: want to plot - pegs left Vs epoch number; visualize convergence.

        # Now, we hopefully converged to a useful policy.
        # Setting epsilon = 0 (full greed)
        # self.run()

    def learning(self):
        # Initial V with some small value
        self.critic.initialize_V()

        # Initial PI, no value associaiton: PI(s,a) = 0
        self.actor.initialize_PI()

        # Run an entire epoch of episodes. i later used for plotting.
        for i in range(self.epochs):
            # Stores states in a list for animation purposes.
            episode = []

            # Reset eligibilites
            self.actor.eligibility = {}
            self.critic.eligibiliy = {}

            # Reset board for this episode.
            self.sim.reset_board()

            # Initial state
            state = self.sim.state
            episode.append(state)

            # Initial action. Action = (state, next_state)
            # TODO: what if there are no possible moves?
            possible_moves = self.sim.actions()
            next_state = self.actor.action_selection(state, possible_moves, 0.2)

            # Algorithm-translation:
            # 1. next_state "=" a/s'
            # 2. state = s
            # 3. next_move = a'

            # Iterate over an episode.
            while (not self.sim.final_state(state)):
                # 1
                # Doing the action, getting a reward. Updating the board.
                reward = self.sim.reward(state, next_state)
                self.sim.set_board_state(next_state)
                # state is then the "previous" state, still.
                episode.append(next_state)

                # 2
                # Actor find next action based on the updated board-state.
                possible_moves = self.sim.actions()
                next_move = self.actor.action_selection(state, possible_moves, 0.2)
                
                # 3
                # Eligibility updated based on "old" state s, and "old action" a
                # e(s,a) <- 1
                self.actor.update_e(state, next_state) # Actor keeps SAP based eligibilites

                # 4
                # Critic critisises
                delta = reward + self.discount * self.critic.evaluate(next_state) - self.critic.evaluate(state)

                # 5
                # Critic eligibility. State based eligibility. e(s) <- 1
                self.critic.update_e(state)
                
                # 6
                # Do learning, based on eligibilities.
                # The last state in episodes, is an action itself.
                for j in range(0, len(episode)-1):
                    # Bookkeeping
                    # TODO: in the final move of an episode, we do not update e for actor.
                    s = episode[j]
                    a = episode[j+1]

                    # (a), (b)
                    # Critic updates value function, based on learning rate and eligibility.
                    self.critic.update_V(s, delta)
                    # Discount the e by gamma, and decay rate.
                    self.critic.update_e(s) # Critic uses state-eligibilities alone.

                    # (c), (d)
                    # Actor updates policy, based on delta. (TD error)
                    self.actor.update_PI(s, a, delta)
                    # Discount the e by gamma, and decay rate. 
                    self.actor.update_e(s, a)

                # Update before proceeding
                state = next_state
                next_state = next_move
                


    # Used for an epsilon-greedy algorithm.
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

