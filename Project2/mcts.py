###
import numpy as np
from collections import defaultdict
import time
###
import hex_board as hb
from copy import deepcopy
from actor import Actor


class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, episolin, root : hb.hex_board, default_policy, exploration_const=1,):
        # Initialize
        self.initialize(root)
        # 
        self.c = exploration_const              # C in UCT
        self.epsilon_0 = episolin               # Initial epsilon value
        #                                       
        self.default_policy = default_policy    # NN actor

    # Initializes / resets the MCTS
    def initialize(self, root : hb.hex_board):
        # Node dictionaries ( ͡° ͜ʖ ͡°)
        self.Q = dict()                         # total reward of each node. key: (state_ID, a)
        self.N = dict()                         # total visit count for each node. key: (state_ID)
        self.N_v = dict()                       # total visit counts for each vertex. Key is (state_ID, a)
        self.children = dict()                  # hold children_IDs of each node_ID
        self.root = deepcopy(root)              # hexboard)

    def M_simulations(self, M : int):
        '''run M simulations to expand the MCT and gather statistics'''
        # TODO: Timelimit instead of number limit
        for i in range(M):
            epsilon = self.epsilon_0 # TODO: Should decrease e as we progress, use i.
            self.single_simulation(i, epsilon)

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

    def single_simulation(self, i, epsilon):
        '''Run a single simulation from root-->leaf-->terminal'''
        '''
        Currently, finds a leaf, exapnds and adds all children, then rollouts from one of them.
        '''
        
        # Root setup
        node = deepcopy(self.root)
        path = [hash(node)]         # Path of current simulation, for backprop
        actions = []                # Actions taken along the way.

        # First time setup: when the search tree consists of only an unexplored root.
        if self.is_leaf(hash(node)): #Root is a leaf, only happens once.
            print('A "Fresh" MCTS is created: empty root.')
            self.N[hash(node)] = 1 # initiate N

        # Tree search algorithms, p1 and p2. Based on the "MCTS for GO" - paper.
        def UCT_1(action):
            '''Tree search for p1, done until we hit a leaf of the tree.'''
            return self.Q[(hash(node), action)] + self.c * np.sqrt(np.log(self.N[hash(node)]) / 1 + self.N_v[(hash(node), action)])
        def UCT_2(action):
            '''Tree search for p2, done until we hit a leaf of the tree.'''
            return self.Q[(hash(node), action)] - self.c * np.sqrt(np.log(self.N[hash(node)]) / 1 + self.N_v[(hash(node), action)])
            
        # iteration variables
        possible_moves = node.possible_moves_pos()

        # Tree search
        while (not self.is_leaf(hash(node))): # Not a leaf
            # In case the game is already over, initiate rollout
            if self.hit_the_bottom(node, path, actions):
                return
            # Use UCT to choose action
            if node.player_turn==1:
                move = max(possible_moves, key=UCT_1) 
            elif node.player_turn==2:
                move = min(possible_moves, key=UCT_2)
            else:
                raise Exception("Node without player turn")

            # Make the move, add to list of actions
            node.make_move(move)
            actions.append(move)

            # Add new node to path
            path.append(hash(node)) # node is altered as we go, need deepcopy
            
            # Update iteration variables
            possible_moves = node.possible_moves_pos()

        # The Leaf can be the final state.
        if self.hit_the_bottom(node, path, actions):
                return
        # At leaf, expand all* children and add them to the tree. * possibly none 
        children_IDs = node.all_children_IDs(possible_moves) # returned in node format (boards)
        self.children[hash(node)] = children_IDs
        self.add_to_tree(hash(node), possible_moves, children_IDs) # First time setup
        
        ##### Rollout on one random child
        # 1 Choose child randomly
        move = node.random_move()
        actions.append(move)

        node.make_move(move)
        path.append(hash(node))
        
        # 2 Rollout, using default algorithm
        while node.game_over == False:
            move_distribution = self.default_policy.move_distribution(node.flatten_state()) # k*k+1 1D list of probabilities.
            legal_moves =  np.multiply(move_distribution, node.possible_moves) # Remove impossible moves.
            norm_moves = legal_moves / np.sum(legal_moves)

            # Epsilon greedy choice TODO: choose based on distribution itself of course!
            z = np.random.uniform(0,1)
            if z > epsilon: # chose highest number
                move_index = np.argmax(norm_moves)              # this is 1D move
                move = (move_index//node.k, move_index%node.k)  # (row, col) of move
            else:
                move = node.random_move()

            # Make the move
            node.make_move(move)

            # Add state-action pair - no need for the rollout
            #actions.append(move)
            #path.append(hash(node))


        # We are now at a final node F.
        if node.winner == 1:
            z = 1
        elif node.winner == 2:
            z = -1
        else:
            # Nash prooved in 1948 that one player has to win.
            raise Exception("No one won!")

        self.backpropagation(path, actions, z)
        #TODO: Seems like UTC is almost deterministic!
        #print("PATH:", path)

    def backpropagation(self, path, actions, z):
        # TODO: might want to backpropagate for the rollout nodes as well.
        for i in range(0, len(actions)):
            # State-actions pairs, node/move.
            # One more node than actions = leaf node
            node_ID = path[i] # Only stores node IDs
            move = actions[i]

            if node_ID==path[-1]:
                print("ACTIONS:", actions)
                print("NODES:", path)
                raise Exception("Nodes:"+str(len(path))+"\tActions:"+str(len(actions))+"\nBackprop loop should stop before leaf node!")
            # Update N, N_v, Q
            self.N[node_ID] += 1
            self.N_v[(node_ID, move)] += 1
            self.Q[(node_ID, move)] += (z - self.Q[(node_ID, move)]) / self.N_v[(node_ID,move)]
        # Update N for leaf node.
        node_ID = path[-1] # The final node
        self.N[node_ID] += 1
       
    def add_to_tree(self, parent_ID, move_list, children_IDs):
        ''' First time setup for a node. Initial values'''
        k = -1
        rand = np.random.uniform(-0.1, 0.1, (len(move_list),))
        for i in range(len(children_IDs)):
            try: # This is a DAG, not a tree. We can find equal states and vertexes from multiple approaches!
                k=self.N_v[(parent_ID, move_list[i])]
            except:
                self.N_v[(parent_ID, move_list[i])] = 1
            try:
                k=self.Q[(parent_ID, move_list[i])]
            except:
                self.Q[(parent_ID, move_list[i])] = rand[i]
            try: 
                k=self.N[children_IDs[i]]
            except:
                 self.N[children_IDs[i]] = 1
        if k==-1:
            return True # All got updated

    def prune_search_tree(self, new_root):
        ''' Prunes search tree by selecting new root '''
        # TODO: remove all unused data from dictinaries. This can be done using children and iterating.
        # Is it worth it?
        self.root = deepcopy(new_root)
     
    def is_leaf(self, node_ID):
        ''' Checks if a node is a leaf node '''
        try: # Check for leaf
            k=len(self.children[node_ID])
            return False
        except:
            return True 

    def hit_the_bottom(self, node, path, actions):
        if node.game_over==True:
            if node.winner == 1:
                z = 1
            elif node.winner == 2:
                z = -1
            self.backpropagation(path, actions, z)
            return True
        return False