'''
RL_engine
'''
# My own modules
from actor import Actor
from critic import Critic
from SW_peg_solitaire import SW, BT
from nn_critic import NN_Critic
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
    def __init__(self, critic_mode, discount, alpha_a, alpha_c, epochs, lambda_a, lambda_c, board_type, board_size, initial_holes, reward_mode):
        '''
        Parameters:
            -> value_mode: mode for value function. 0: dictionary, 1: NN
            ->
        '''
        # NN parameters
        self.n_layers = 1
        self.input_size = board_size**2 #TODO: function to calculte this

        # RL parameters
        self.critic_mode = critic_mode
        self.epochs = epochs
        self.discount = discount
        self.reward_mode = reward_mode

        # The two players
        self.actor = Actor(discount, alpha_a, lambda_a)
        if critic_mode == 1:
            self.critic = Critic(discount, alpha_c, lambda_c)
        else:
           self.critic = NN_Critic(discount, alpha_c, lambda_c, self.n_layers, self.input_size)
        
        
        # The board used for simulations field
        self.sim = SW(board_type, board_size, initial_holes)
        #print("Board to learn:\n", self.sim.state)

    # Learning 
    def learning(self, e_0, epsilon_mode):
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
            # Print progress
            #if i%(int(0.5*self.epochs/10)) == 0:
            #    print("Learning progress: "+str(100*i/self.epochs)+"%")
  
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
            next_state = self.actor.action_selection(state, possible_moves, self.epsilon(epsilon_mode, e_0, episode_nr=i, epochs=self.epochs))

            # Iterate over an episode.
            while (not self.sim.final_state(state)):
                # 1
                # Doing the action, getting a reward, updating the board.
                reward = self.sim.reward(state, next_state, self.reward_mode)
                self.sim.set_board_state(next_state)
                
                # 2
                # Actor find next action based on the updated board-state.
                possible_moves = self.sim.child_states(next_state)
                next_move = self.actor.action_selection(next_state, possible_moves, self.epsilon(epsilon_mode, e_0, episode_nr=i, epochs=self.epochs))
                
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
    
    
    ## NN learning
    def nn_learning(self, e_0, epsilon_mode):
        # Initial net V.
        self.critic.initialize_V()

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
            # Print progress
            if i%(int(0.5*self.epochs/10)) == 0:
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
            next_state = self.actor.action_selection(state, possible_moves, self.epsilon(epsilon_mode, e_0, episode_nr=i, epochs=self.epochs))

            # Iterate over an episode.
            state_target = []
            while (not self.sim.final_state(state)):
                # 1
                # Doing the action, getting a reward, updating the board.
                reward = self.sim.reward(state, next_state, self.reward_mode)
                self.sim.set_board_state(next_state)
                
                # 2
                # Actor find next action based on the updated board-state.
                possible_moves = self.sim.child_states(next_state)
                next_move = self.actor.action_selection(next_state, possible_moves, self.epsilon(epsilon_mode, e_0, episode_nr=i, epochs=self.epochs))
                
                # 3
                # Eligibility updated based on "old" state s, and "old" action a. e(s,a) <- 1
                self.actor.update_e(state, next_state, 1) # Actor keeps SAP based eligibilites

                # 4
                # Critic critisises. Using NN to predict, and reward + NN to make a target.
                state_input = self.state_to_input(state, self.sim.board_size, self.sim.board_type)
                next_state_input = self.state_to_input(next_state, self.sim.board_size, self.sim.board_type)
                # Model predictions
                V_theta = self.critic.split_model.model.predict(state_input)
                V_star = reward + self.discount*self.critic.split_model.model.predict(next_state_input)
                # Save all state-targets for later use (state stored in input-format)
                state_target.append((state_input, V_star))
                # TD error for this state-pair
                delta = self.critic.update_delta(V_star, V_theta)

                # 5 - Eligibilites are different for critic now. (e(s) <- 1)
                # They are updated and kept in "update gradient" function.
                
                # 6
                # Do learning, based on eligibilities.
                episode.append(next_state) # Add a to the list.
                for j in range(0, len(episode)-1):
                    # Bookkeeping
                    s = episode[j]
                    a = episode[j+1] # s'

                    # (c), (d)
                    # Actor updates policy, based on delta. (TD error)
                    self.actor.update_PI(s, a, delta)
                    # Discount the e by gamma, and decay rate. 
                    self.actor.update_e(s, a, 2)

                # 6.5
                # We do the learning for the critic
                for j in range(len(state_target)):
                    # Do learning
                    # Takes care of eligibilities and all.
                    self.critic.update_V(state_target[j][0], state_target[j][1])

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


    # Demonstrating
    def run(self):
        '''
        Runs on the completed PI. Animates the solution.
        '''
        # Saves the episode for animation.
        episode = []
        self.sim.reset_board()
        # Initial state
        state = self.sim.state
        episode.append(state)

        while (not self.sim.final_state(state)):
            # Look for actions
            possible_moves = self.sim.child_states(state)
            # Determine first action.
            state = self.actor.action_selection(state, possible_moves, self.epsilon(0, 0, 0))
            episode.append(state)
        
        
        return episode
    # Calculating epsilon for the epsilon-greedy algorithm.
    def epsilon(self, mode, e_0, episode_nr=0, epochs=0):
        '''
        Mode 0 (default): For full greed.
        
        Mode 1: fixed constant for big exploration, and thus learning.

        Mode 2: Linear decent

        Mode 3: logarithmic decay / inverse sigmoid

        Mode 4: ReLu-like behaviour
        '''
        # Mode 1
        if mode == 1:
            return 0.3
        
        # Mode 2
        if mode == 2:
            return (e_0 - e_0*episode_nr/epochs)
        
        if mode == 3:
            if (episode_nr/epochs) <= 0.5:
                return e_0
            else:
                return (e_0 - e_0*(episode_nr - epochs*0.5)/(epochs*0.5))
        # Mode 0

        if mode == 4:
            if (episode_nr/epochs) < 0.9:
                return (e_0 - e_0*episode_nr/(epochs*0.9))
            else:
                return 0
        return 0
    # Make states into input for 
    def state_to_input(self, state, board_size, board_type):
        ''' Turns the size*size matrix into an input'''
       
        if board_type == BT.DIAMOND:
            i = 0
            inputs = np.zeros(board_size*board_size)
            for row in range(board_size):
                for col in range(board_size):
                    inputs[i] = state[row][col]
                    i+=1

        if board_size == BT.TRIANGLE:
            # Keeping track of layers of triangle
            n_in_layer = 1 # Number of nodes in this layer/row number
            n_l_counter = 1 #Current position within layer/col number
            size = 0
            for i in range(1, board_size+1):
                size += i
            inputs = np.zeros(size)

            for i in range(size):
                if n_l_counter > n_in_layer:
                    # One more node per layer
                    # We are in next layer
                    n_in_layer += 1
                    n_l_counter = 1
                inputs[i] = state[n_in_layer-1 ][n_l_counter-1]
                n_l_counter += 1

        return inputs