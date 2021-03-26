###
import numpy as np
from collections import defaultdict

###
import hex_board as hb
from copy import deepcopy
from actor import Actor


class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, episolin, board : hb.hex_board, default_policy : Actor, exploration_const=1,):
        # Node dictionaries ( ͡° ͜ʖ ͡°)
        self.Q = dict()                         # total reward of each node. key: (state, a)
        self.N = dict()                         # total visit count for each node. key: (state)
        self.N_v = dict()                       # total visit counts for each vertex. Key is (state, a)
        self.children = dict()                  # children of each node
        
        # 
        self.c = exploration_const              # C in UCT
        self.epsilon_0 = episolin                 # Initial epsilon value

        #                                       
        self.root = board                       # np.array((k,k)), empty
        self.default_policy = default_policy    # NN actor

    def M_simulations(self, M : int):
        '''run M simulations to expand the MCT and gather statistics'''
        # TODO: Timelimit instead of number limit
        for i in range(M):
            epsilon = self.epsilon_0 # TODO: Should decrease as we progress somehow.
            self.single_simulation(i, epsilon)

    def single_simulation(self, i, epsilon):
        '''Run a single simulation from root-->leaf-->terminal'''
        '''
        Currently, finds a leaf, exapnds and adds all children, then rollouts from one of them.
        '''
        # First time setup
        node = self.root
        path = [deepcopy(node)] # Path of current simulation, for backprop
        actions = []            # Actions taken along the way.



        # Tree search algorithms
        def UCT_1(action):
            '''Tree search for p1, done ntil we hit a leaf of the tree.'''
            return self.Q[(node, action)] + self.c * np.sqrt(np.log(self.N[node]) / 1 + self.N_v[(node, action)])
        
        def UCT_2(action):
            '''Tree search for p2, done ntil we hit a leaf of the tree.'''
            return self.Q[(node, action)] - self.c * np.sqrt(np.log(self.N[node]) / 1 + self.N_v[(node, action)])

        # iteration variables
        possible_moves = node.possible_moves_pos()
        no_possible_children = len(possible_moves)
        no_current_children = len(self.children[node])
        
        # Tree search
        while no_current_children == no_possible_children and no_possible_children != 0: # Not a leaf or a terminal node 
            # Use UCT to choose action
            if node.player_turn==1:
                move = max(possible_moves, key=UCT_1) # TODO:!! Will crash for root, as UCT variables are uninitiated.
            if node.player_turn==2:
                move = min(possible_moves, key=UCT_2)
            else:
                raise Exception("Node without player turn")

            # make the move 
            node.make_move(move)
            actions.append(move)

            # Add new node to path
            path.append(deepcopy(node)) # node is altered as we go, need deepcopy
            
            # update iteration variables
            possible_moves = node.possible_moves_pos()
            no_possible_children = len(possible_moves)
            no_current_children = len(self.children[node])

        # At leaf, expand !all! children and add them to the tree. 
        children_nodes = node.all_children_boards() # returned in node format (boards)
        self.children[node] = children_nodes
        
        ##### Rollout on one random child
        # 1 Chose child randomly
        move = node.random_move()
        actions.append(move)

        node.make_move(move)
        path.append(deepcopy(node))
        
        # 2 Rollout, using default algorithm
        while node.game_over == False:
            move_distribution = self.default_policy(node.flatten_state()) # k*k 1D list of probabilities.
            legal_moves =  np.multiply(move_distribution, node.possible_moves) # Remove impossible moves.
            norm_moves = legal_moves / np.sum(legal_moves)
            
            # Epsilon greedy choice TODO: choose based on distribution itself of course!
            z = np.random.randint(0,1)
            if z > epsilon: # chose highest number
                move_index = np.argmax(norm_moves)              # this is 1D move
                move = (move_index//node.k, move_index%node.k)  # (row, col) of move
            else:
                move = node.random_move()

            # Make the move, add new node to path
            node.make_move(move)

            actions.append(move)
            path.append(deepcopy(node))

        # We are now at a final node F.
        if node.winner == 1:
            z = 1
        elif node.winner == 2:
            z = -1
        else:
            raise Exception("No one won!")

        self.backpropagation(path, actions, z)


    def backpropagation(self, path, actions, z):
        # TODO: might not want to backpropagate for the rollout nodes.
        for i in range(0, len(path)):
            node = path[i]
            move = actions[i]

            try:
                self.N[node] += 1
            except:
                self.N[node] = 1
            
            try:
                self.N_v[(node,move)] += 1
            except:
                self.N_v[(node,move)] = 1

            try:
                self.Q[(node, move)] += (z - self.Q[(node, move)]) / self.N_v[(node,move)]
            except:
                self.Q[(node, move)] = (z) / self.N_v[(node,move)]
