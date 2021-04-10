###
import numpy as np
from collections import defaultdict
import time
###
import hex_board as hb
from copy import deepcopy
from actor import Actor


# New MCTS, based on book.
class MCTS:
    "Monte Carlo tree searcher based on paper. First rollout the tree then choose a move."
    # Init. function for a MCTS instance.
    def __init__(self, episolin, root : hb.hex_board, default_policy, exploration_const=1,):
        # Initialize
        self.initialize(root)
        # 
        self.c = exploration_const              # C in UCT
        self.epsilon_0 = episolin               # Initial epsilon value
        #                                       
        self.default_policy = default_policy    # NN actor
    # Resets the MCTS. Removing the tree entirely.
    def initialize(self, root : hb.hex_board):
        # Node dictionaries ( ͡° ͜ʖ ͡°)
        self.Q = dict()                         # total reward of each node. key: (state_ID, a)
        self.N = dict()                         # total visit count for each node. key: (state_ID)
        self.N_v = dict()                       # total visit counts for each vertex. Key is (state_ID, a)
        self.root = deepcopy(root)              # hexboard)
    # Run single simulations for a time, building the tree.
    def simulate_timed(self, s : int, progress : float, verbose=False):
        ''' Runs simulations until time is out. s is seconds'''
        start_time = time.time()
        current_time = time.time()
        elapsed_time = current_time-start_time
        i = 0
        while elapsed_time < s:
            epsilon = self.epsilon_0 * (1-progress) # Gradual decay from epsilon_0 -> 0, based on progress in *epoch*.
            self.single_simulation(i, epsilon)      
            i+=1
            current_time = time.time()
            elapsed_time = current_time-start_time
        
        if verbose==True:
            print("You manged to do",i,"simulations in",s,"seconds.")
    # Run a single simulation from root to terminal node.
    def single_simulation(self, i, epsilon):
        '''Run a single simulation from root-->leaf-->terminal'''
        # 1 Reset node to root.
        node = deepcopy(self.root)

        # 2 Path of current sim and actions taken.
        path = []
        actions = []
        
        # 3 Search R-->L
        while not node.game_over:
            # Add node to path. NB! We don't add T node.
            path.append(hash(node))
            if hash(node) in self.N.keys():  # Is in tree
                # Select move UCT style
                move = self.tree_move(node)
                # Make move.
                node.make_move(move)
                actions.append(move)

            else: # Not in tree
                self.add_node_to_tree(node)
                break
        
        # 4 We are now in the newly added leaf L. 
        # In case of L =/= T, actions has one less element than path, need one more action.
        if not node.game_over:
            move = self.default_move(node, epsilon)
            # Make move
            node.make_move(move)
            actions.append(move)

        # 5 Keep using default algorithm, until we hit T.
        while not node.game_over:
            # Select move
            move = self.default_move(node, epsilon)
            # Make move
            node.make_move(move)

        # 6 Now we are in a final state T. 
        z = self.winner_of_game(node)
        self.backpropagation(path, actions, z)
    # Do a tree move based on UCT algorithm.
    def tree_move(self, node : hb.hex_board):
        '''Uses UCT algorithm to select a move'''
        # All legal actions
        legal = node.possible_moves_pos()
        node_ID = hash(node)
        # Tree search algorithms, p1 and p2. Based on the "MCTS for GO" - paper.
        
        def UCT_1(action):
            '''Tree search for p1, done until we hit a leaf of the tree.'''
            return self.Q[(node_ID, action)] + self.c * np.sqrt(np.log(self.N[node_ID]) / (self.N_v[(node_ID, action)]))
        def UCT_2(action):
            '''Tree search for p2, done until we hit a leaf of the tree.'''
            return self.Q[(node_ID, action)] - self.c * np.sqrt(np.log(self.N[node_ID]) / (self.N_v[(node_ID, action)]))

        if node.player_turn==1:
                move = max(legal, key=UCT_1) 
        elif node.player_turn==-1:
            move = min(legal, key=UCT_2)
        else:
            raise Exception("Node without player turn")

        return move
    # Do a default move, based on default algorithm
    def default_move(self, node : hb.hex_board, epsilon : float):
        '''Uses default algorithm to chose a move from node'''
        # Epsilon greedy choice TODO: choose based on distribution itself of course! np.random.choice
        z = np.random.uniform(0,1)
        if z > epsilon:     # Chose highest number, greedy.
            # Get and normalize moves from the default policy.
            move_distribution = self.default_policy.move_distribution(node.flatten_state()) 
            legal_moves =  np.multiply(move_distribution, node.possible_moves) # Remove impossible moves.
            # Problem: If all legal moves are picked with 0% chance from the move-distribution => zero array.
            # This happens if the NN is very confident it doesnt want to pick that action.
            if np.sum(legal_moves) == 0:
                return node.random_move() # Odd case, just return random.
            # normalize moves
            norm_moves = legal_moves / np.sum(legal_moves)
            # Pick best move, greedily.
            move_index = np.argmax(norm_moves)              # 
            move = (move_index//node.k, move_index%node.k)  # 
        else:               # Choose random move.
            move = node.random_move()
        
        return move
    # Add a new node to the tree.
    def add_node_to_tree(self, node):
        '''Adds a single node to the tree''' 
        node_ID = hash(node)
        self.N[node_ID] = 0
        actions = node.possible_moves_pos()
        for a in actions:
            self.Q[(node_ID, a)] = 0
            self.N_v[(node_ID, a)] = 1 # To avoid dividing by zero in UCT.
    # Backpropagate and update tree-nodes.
    def backpropagation(self, path, actions, z):
        for i in range(0, len(actions)):
            # Element extraction
            node_ID = path[i] # Only stores node IDs
            move = actions[i]
            # Update N, N_v, Q
            self.N[node_ID] += 1
            self.N_v[(node_ID, move)] += 1
            self.Q[(node_ID, move)] += (z-self.Q[(node_ID, move)]) / self.N_v[(node_ID,move)]
    # Prune the tree, setting a new root.
    def prune_search_tree(self, new_root):
        ''' Prunes search tree by selecting new root '''
        # Is it worth it?
        self.root = deepcopy(new_root)
    # Check who won, return the z value.
    def winner_of_game(self, node):
        if node.game_over==True:
            if node.winner == 1:
                return 1 # z value
            elif node.winner == -1:
                return -1 # z value
            else:
                raise Exception("Aint no winner!")
        else:
            raise Exception("Game aint over!")