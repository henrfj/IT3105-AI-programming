import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
# My own modules
from SW_peg_solitaire import BT

class Display:
    '''
    Class for drawing fine graphs (node-diagrams), displaying the game board.
    Must be able to draw both triangle and diamond-shaped graphs,
    and nodes of several colours. 
    '''

    def __init__(self): 
        self.graph = nx.Graph()


    def display_board(self, state, board_type, board_size):
        '''
        Instead of the state we are used to, 
        this method takes the array / integer version.
        '''
        # Produce a neighbourhood matrix
        neighbour_matrix = self.transform(state, board_type, board_size)
        # Use networkX to make a graph based on the neighbourhood matrix
        self.graph = nx.from_numpy_array(neighbour_matrix)
        # Color nodes according to state matrix
        color_map = self.color_nodes(state, board_type, board_size)

        plt.figure(figsize=(10,10))
        nx.draw(self.graph, node_color=color_map, with_labels=True)

        #nx.draw(self.graph, with_labels=True)
        plt.show()

    def transform(self, state, board_type, board_size):
        '''
        Takes a integer array representation of the board, and transform it into a adjacency matrix.
        '''
        
        if board_type == BT.DIAMOND:
            adjacency = np.zeros((board_size*board_size, board_size*board_size), int)
            for row in range(board_size**2):
                
                if ((row + 1) < board_size**2) and ((row+1)%board_size) != 0:
                    # 1's above the diagonal
                    adjacency[row][row+1] = 1
                    # Diagonal reflection
                    adjacency[row+1][row] = 1
                if ((row+board_size-1) < board_size**2) and ((row%board_size) != 0):
                    # 1's above diagonal
                    adjacency[row][row+board_size-1] = 1
                    # Diagonal reflection
                    adjacency[row+board_size-1][row] = 1
                if row+board_size < board_size**2:
                    # 1's above diagonal
                    adjacency[row][row+board_size] = 1
                    # Diagonal reflection
                    adjacency[row+board_size][row] = 1
                    
        
        elif board_type == BT.TRIANGLE:
            
            # Calculate number of nodes=size of adjacency matrix
            size = 0
            for i in range(1, board_size+1):
                size += i
            adjacency = np.zeros((size,size), int)

            # Keeping track of layers of triangle
            n_in_layer = 1 #Number of nodes in this layer
            n_l_counter = 1 #Current position within layer

            # Iterate through all rows and add nodes
            for row in range(size):
                if n_l_counter > n_in_layer:
                    # One more node per layer
                    # We are in next layer
                    n_in_layer += 1
                    n_l_counter = 1

                if (n_l_counter != n_in_layer) and (row+1 < size) :
                    # Not on the last node in the layer
                    # and not on the last node
                    adjacency[row][row+1] = 1
                    # Diagonal reflection
                    adjacency[row+1][row] = 1

                if ((row + n_in_layer) < size) and ((row + n_in_layer + 1) < size):
                    # This rule holds for all except for the last layer.
                    adjacency[row][row + n_in_layer] = 1
                    adjacency[row][row + n_in_layer + 1] = 1
                    # Diagonal reflection
                    adjacency[row + n_in_layer][row] = 1
                    adjacency[row + n_in_layer + 1][row] = 1


                n_l_counter += 1


        return adjacency


    def color_nodes(self, state, board_type, board_size): 
        '''
        Colors the nodes accordingly.
        black: peg, white: hole
        '''
        color_map = []
        
        if board_type == BT.DIAMOND:
            # Iterator
            i = 0
            for i in range(board_size*board_size):
                row = i // board_size
                col = i - row*board_size
                if state[row][col] == 1:
                    # Its a peg
                    color_map.append("black")
                
                elif state[row][col] == 0:
                    # Its a hole
                    color_map.append("white")

                i+=1

        elif board_type == BT.TRIANGLE:
            # Keeping track of layers of triangle
            n_in_layer = 1 # Number of nodes in this layer/row number
            n_l_counter = 1 #Current position within layer/col number
            
            size = 0
            for i in range(1, board_size+1):
                size += i

            for i in range(size):
                if n_l_counter > n_in_layer:
                    # One more node per layer
                    # We are in next layer
                    n_in_layer += 1
                    n_l_counter = 1
                
                if state[n_in_layer-1][n_l_counter - 1] == 1:
                    # Its a Peg
                    color_map.append("black")
                elif state[n_in_layer-1][n_l_counter - 1] == 0:
                    # Its a hole
                    color_map.append("white")

                n_l_counter += 1

        return color_map


    def node_pos(self, state, board_type, board_size):
        nodepos={}
        if board_type == BT.DIAMOND:
            # Iterator, node number
            i = 0
            for i in range(board_size*board_size):
                row = i // board_size
                col = i - row*board_size
                
                nodepos[i] = (row, col)
                    
                i+=1

        elif board_type == BT.TRIANGLE:
            # Keeping track of layers of triangle
            n_in_layer = 1 # Number of nodes in this layer/row number
            n_l_counter = 1 #Current position within layer/col number
            
            size = 0
            for i in range(1, board_size+1):
                size += i

            for i in range(size):
                if n_l_counter > n_in_layer:
                    # One more node per layer
                    # We are in next layer
                    n_in_layer += 1
                    n_l_counter = 1
                
                nodepos[i] = (n_in_layer-1, n_l_counter-1)

                n_l_counter += 1

        return nodepos
