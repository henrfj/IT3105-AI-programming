import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
# My own modules
from board import BT

class Display:
    '''
    Class for drawing fine graphs (node-diagrams), displaying the game board.
    Must be able to draw both triangle and diamond-shaped graphs,
    and nodes of several colours. 
    '''

    def __init__(self, spacing): 
        self.graph = nx.Graph()
        self.spacing = spacing

    def display_board(self, state, board_type, board_size):
        '''
        Instead of the state we are used to, 
        this method takes the array / integer version.
        '''
        neighbour_matrix = self.transform(state, board_type, board_size)
        print(neighbour_matrix)
        self.graph = nx.from_numpy_array(neighbour_matrix)
        nx.draw(self.graph)
        plt.show()

    def transform(self, state, board_type, board_size):
        '''
        Takes a integer array representation of the board, and transform it into a adjacency matrix.
        '''
        adjacency = np.zeros((board_size*board_size, board_size*board_size), int)
        if board_type == BT.DIAMOND:
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
            raise Exception("Not implemented yet, you fool")
        
        return adjacency


    def color_nodes(self, arr_state, board_type, board_size): ## function that creates nodes and edges.
        pass
    
    

