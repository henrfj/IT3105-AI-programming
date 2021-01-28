'''
The board class.

'''
import numpy as np
from node import *
# To add additional starting-holes
import random as rd


# Enum  for easier to read code
from enum import Enum
class BT(Enum):
    DIAMOND = 0
    TRIANGLE = 1


class Board:
    ''' 
    Board class. Initializing and keeping track of board state.
    '''

    def __init__(self, board_type: BT, size: int, initial_holes: int):
        # Setting size of board
        self.size = size
        # Setting board type
        if (board_type is BT.DIAMOND):
            self.state = self.diamond_board()
        elif (board_type is BT.TRIANGLE):
            self.state = self.triangle_board()
        else:
            print("Board must be Diamond or triangle")
            print("Defaulting to Diamond board")
            self.state = self.diamond_board()
        # Emptying the starting-holes in the board 
        self.initialize_board(initial_holes)
        

    def diamond_board(self):
        '''
        Generate a size*size list with nodes.
        Nodes are initially filled with pegs
        '''
        # Create a empty (zero-filled) size x size matrix, able to contain Holes
        board = np.zeros((self.size, self.size), Node)
        # Fill the board with default Node, being filled with pegs.
        for row in range(0,self.size):
            for col in range(0, self.size):
                board[row][col] = Node()

        return board

    def triangle_board(self):
        '''
        Generate a size*size list with holes.
        Holes are initially filled with pegs.
        Doesn't fill unused part of board.
        '''
        board = [[]]
        # TODO: Implement this function
        
        return board

    def initialize_board(self, initial_holes):
        '''
        Alters the state, to contain #intial_holes holes,
        ready to start playing
        '''
        # Number of holes inserted
        n = 1
        # Initial hole created close to the center
        self.state[(self.size//2) - 1][(self.size//2) - 1].set_status(Status.EMPTY)

        # Crude, but doing the job
        while n < initial_holes:
            row = rd.randint(0, self.size-1)
            col = rd.randint(0, self.size-1)

            if (self.state[row][col].get_status() != Status.EMPTY):
                self.state[row][col].set_status(Status.EMPTY)
                n+=1