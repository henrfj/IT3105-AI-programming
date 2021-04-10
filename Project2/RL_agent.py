'''RL agent, using MCTS RL combined with NN actor'''

# 
import numpy as np
import tensorflow.keras as ker

# Own modules
from actor import Actor, Random_actor
from mcts import MCTS
from hex_board import hex_board as hb

class Agent:
    '''The MCTS RL agent'''

    def __init__(self, alpha : float, layer_dimensions : list, k : int, epsilon : float, s : int, RBUF_size : int, mbs : int, mode=2, activation="tanh", optimizer=ker.optimizers.SGD):
        # Params.
        self.s = s                              # Seconds to run a MCTS simulation before choosing move.
        self.k = k                              # Board size. 
        self.input_size = int(k**2) + 1         # Adds one for the player ID
        self.RBUF_size = RBUF_size              # typically 512
        self.mbs = mbs                          # Minibatch size, needs to be smaller than the RBUF size ofc.
        
        # Modules.
        self.actor = Actor(alpha, layer_dimensions, self.input_size, activation=activation, optimizer=optimizer)
        self.board = hb(k)                                              # Used for the actual games, and passed into mcts for sims.
        if mode == 2:
            print("---------Using the NN actor---------")
            self.mcts = MCTS(epsilon, self.board, self.actor)           
        else:
            print("---------Using random actor---------")
            self.mcts = MCTS(epsilon, self.board, Random_actor(self.k)) # For generating training data.

    def random_training_sim(self, actual_games : int, interval : int, epochs : int,  verbose=False):
        '''Run an actual_games-number of episodes. This uses a random mcts actor, and is used to debug not for actual demo.
        Fills the entire RBUF, and then trains hard (10000 epochs), to fit this data best as possible.'''
        # 1
        # Interval = actual games performed between each saved NN.
        
        # 2
        ## Initialize replay buffer, made big enough
        feature_BUF = np.empty((self.RBUF_size, (self.k**2)+1), dtype=float)   # For input states (with ID)
        target_BUF = np.empty((self.RBUF_size, (self.k**2)), dtype=float)      # For target distributions, D
        q = 0 # Index of least recently added case.

        # 3
        # Initialize actor, already done - suppose each agent only has one actor: 1-1
        starting_player = 1 # ! Alternate who starts !

        # 4
        # Run the number of actual games, dictated by actual_games
        for i in range(actual_games):
            if verbose==True:
                print("Progress: "+str(i*100/actual_games)+"%")
            # a, b Alternate starting player
            if starting_player == 1:
                starting_player = -1
            else:
                starting_player = 1
            # c, d
            # Initialize empty game board, reset MCTS. ¯\_(ツ)_/¯ ( ͡° ͜ʖ ͡°)
            self.board.initialize_states(starting_player=starting_player)
            self.mcts.initialize(self.board)

            # d
            # Run one actual game
            while not self.board.game_over and q < self.RBUF_size:
                # Prune MCTS tree so that root = board state. Does nothing the first move.
                self.mcts.prune_search_tree(self.board)
                
                # Run a new sim
                self.mcts.simulate_timed(self.s, progress=(i/actual_games))
                all_moves = self.board.possible_moves_pos_2()           
                
                # Find best move - by visit counts
                D = self.get_the_D(hash(self.board), all_moves, self.mcts.N_v) # normalized visit counts.
                
                # Store (s,D) in replay buffer.
                feature_BUF[q] = self.board.flatten_state()
                target_BUF[q] = D
                q+=1

                # Apply best move
                best_move = all_moves[np.argmax(D)] # The highest index of D, is the best move.
                self.board.make_move(best_move)
                
                
        
        ## e Train based on the last couple of cases.
        print("======================\nTRAINING\n======================")
        # TODO: Actually use minibatchsize, not send all!
        self.actor.train(feature_BUF[:q], target_BUF[:q], mbs=q, epochs=epochs)
        
        if q == self.RBUF_size:
            print("The buffer is filled up!")
        else:
            print("Using what we got...")
        
        print("Saving model to disk.")
        self.actor.model.save("./power_model_3")
        print("Saving data to disk.")
        ## save to csv file
        np.savetxt('feature_BUF.csv', feature_BUF, delimiter=',')
        np.savetxt('target_BUF.csv', target_BUF, delimiter=',')
    
    def NN_training_sim(self, actual_games : int, interval : int, epochs : int,  verbose=False, save_model=False):
        '''Run an actual_games-number of episodes, uses the NN actor and trains underways.
        If self.mbs = self.RBUF_size we just use the most recent to train.
        '''
        
        # 1, 2
        ## Initialize replay buffer, made big enough
        model_paths = []
        feature_BUF = np.empty((self.RBUF_size, (self.k**2)+1), dtype=float)   # For input states (with ID)
        target_BUF = np.empty((self.RBUF_size, (self.k**2)), dtype=float)      # For target distributions, D
        q = 0 # Index of least recently added case.
        wrap_around = False # Have we filled RBUF?

        # 3
        # Initialize actor, already done - suppose each agent only has one actor: 1-1
        starting_player = 1

        # 4
        # Run the number of actual games, dictated by actual_games
        for i in range(actual_games):
            if verbose==True:
                print("Progress: "+str(i*100/actual_games)+"%")
            # a, b Change starting player
            if starting_player == 1:
                starting_player = -1
            else:
                starting_player = 1
            # c, d
            # Initialize empty game board, reset MCTS. ¯\_(ツ)_/¯ ( ͡° ͜ʖ ͡°)
            self.board.initialize_states(starting_player=starting_player)
            self.mcts.initialize(self.board)
            

            # d
            # Run one actual game
            while not self.board.game_over:
                # Prune MCTS tree so that root = board state. Does nothing the first move.
                self.mcts.prune_search_tree(self.board)
                
                # Run a new sim
                self.mcts.simulate_timed(self.s, progress=(i/actual_games))
                all_moves = self.board.possible_moves_pos_2()           
                
                # Find best move - by visit counts
                D = self.get_the_D(hash(self.board), all_moves, self.mcts.N_v) # normalized visit counts.
                
                # Store (s,D) in replay buffer.
                if q == self.RBUF_size:
                    q = 0
                    wrap_around = True

                feature_BUF[q] = self.board.flatten_state()
                target_BUF[q] = D
                q+=1

                # Apply best move
                best_move = all_moves[np.argmax(D)] # The highest index of D, is the best move.
                self.board.make_move(best_move)
                
                
            ## e Train based on the last couple of cases.
            print("======================\nTRAINING\n======================")
            if wrap_around and self.mbs==self.RBUF_size:
                print("Training on all the most recent data.")
                self.actor.train(feature_BUF, target_BUF, mbs=self.mbs, epochs=epochs) # Assuming RBUF size = mbs!

            elif (q >= self.mbs or wrap_around) and (self.mbs < self.RBUF_size):
                print("Training on minibatches of data.")
                if wrap_around:
                    self.actor.train(feature_BUF, target_BUF, mbs=self.mbs, epochs=epochs)
                else: # Not wrapped around yet
                    self.actor.train(feature_BUF[:q], target_BUF[:q], mbs=self.mbs, epochs=epochs)
            else:
                print("Not enough data to train yet.")

            ## f Save model for TOPP. Also saves the untrained net.
            if (i==0 or (i+1)%interval == 0) and (save_model):
                print("Saving model to disk.")
                if i != 0:
                    print("This is done at episode no.",i+1)
                    model_paths.append("./demo_model_"+str((i+1)*100/actual_games))
                    self.actor.model.save("./demo_model_"+str((i+1)*100/actual_games))
                else:
                    print("This is done at episode no.",i)
                    model_paths.append("./demo_model_"+str((i)*100/actual_games))
                    self.actor.model.save("./demo_model_"+str((i)*100/actual_games))

        #self.actor.model.save("./iron_man_mk_2")    
        return model_paths

    def get_the_D(self, node_ID, all_moves, visitation_count):
        '''Returns the D array of normalized visitation count, to be used as target in training.
            all_moves: a list of -1 where it is impossible, and (row,col) where it is possible to move.
            visitation_count: the number of times each possible moves was executed; ordered in same order ass all_moves'''
        D = np.zeros((self.k**2,))

        for i in range(self.k**2):
            if all_moves[i] != -1: # Its one of the legal moves.
                move = all_moves[i]
                try:
                    D[i] = visitation_count[(node_ID, move)]  # If the move has been visited.
                except:
                    D[i] = 0                                  # If it hasn't been visited.

        # Normalize and return the D
        D = D / np.sum(D)
        #return D
        return D

    def gen_random_minibatch(self, inputs, targets, mbs):
        '''Generate random minibatch of size mbs'''
        indices = np.random.randint(len(inputs), size=mbs)
        return inputs[indices], targets[indices]

                