'''RL agent, using MCTS RL combined with NN actor'''

# 
import numpy as np

# Own modules
from actor import Actor, Random_actor
from mcts import MCTS, BOOK_MCTS
from hex_board import hex_board as hb

class Agent:
    '''The MCTS RL agent'''

    def __init__(self, alpha : float, layer_dimensions : list, k : int, epsilon : float, s : int, RBUF_size : int, mbs):
        # Params.
        self.s = s                              # Seconds to run a MCTS simulation before choosing move.
        self.k = k                              # Board size. 
        self.input_size = int(k**2) + 1         # Adds one for the player ID
        self.RBUF_size = RBUF_size              # typically 512
        self.mbs = mbs                          # Minibatch size, needs to be smaller than the RBUF size ofc.
        
        # Modules.
        self.actor = Actor(alpha, layer_dimensions, self.input_size)
        self.board = hb(k)                                          # Used for the actual games, and passed into mcts for sims.
        #self.mcts = BOOK_MCTS(epsilon, self.board, self.actor)      # For generating training data.
        self.mcts = BOOK_MCTS(epsilon, self.board, Random_actor(self.k))

    def run_training_sim(self, actual_games : int, interval : int, epochs : int,  verbose=False):
        '''Run an actual_games-number of episodes'''
        # 1
        # Interval = actual games performed between each saved NN.
        
        # 2
        ## Initialize replay buffer, made big enough
        feature_BUF = np.empty((self.RBUF_size, (self.k**2)+1), dtype=float)   # For input states (with ID)
        target_BUF = np.empty((self.RBUF_size, (self.k**2)), dtype=float)      # For target distributions, D
        q = 0 # Index of least recently added case.

        # 3
        # Initialize actor, already done - suppose each agent only has one actor: 1-1

        # 4
        # Run the number of actual games, dictated by actual_games
        for i in range(actual_games):
            if verbose==True:
                print("Progress: "+str(i*100/actual_games)+"%")
            # a, b, c, d
            # Initialize empty game board, reset MCTS. ¯\_(ツ)_/¯ ( ͡° ͜ʖ ͡°)
            self.board.initialize_states()
            self.mcts.initialize(self.board)
            

            # d
            # Run one actual game
            while not self.board.game_over or q==self.RBUF_size:
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
            #print("======================\nTRAINING\n======================")
            #self.actor.train(feature_BUF[:q], target_BUF[:q], mbs=q, epochs=epochs)
        
        ## e Train based on the last couple of cases.
        print("======================\nTRAINING\n======================")
        # TODO: Actually use minibatchsize, not send all!
        self.actor.train(feature_BUF[:q], target_BUF[:q], mbs=q, epochs=epochs)
        
        if q == self.RBUF_size:
            print("The buffer is filled up!")
        else:
            print("Using what we got...")
        
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

                