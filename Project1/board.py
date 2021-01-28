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
        # Setting size and type of board
        self.size = size
        self.type = board_type
        # Setting board type
        if (self.type is BT.DIAMOND):
            self.state = self.diamond_board(initial_holes)
        elif (self.type is BT.TRIANGLE):
            self.state = self.triangle_board(initial_holes)
        else:
            print("Board must be Diamond or triangle")
            print("Defaulting to Diamond board")
            self.state = self.diamond_board(initial_holes)  
        

    def diamond_board(self, initial_holes):
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

        '''Now initialize the holes'''
        # Number of holes inserted. (Initial_holes >= 1)
        n = 1
        # Initial hole created close to the center
        board[(self.size//2)][(self.size//2)].set_status(Status.EMPTY)

        # Crude, but doing the job
        while n < initial_holes:
            row = rd.randint(0, self.size-1)
            col = rd.randint(0, self.size-1)

            if (board[row][col].get_status() != Status.EMPTY):
                board[row][col].set_status(Status.EMPTY)
                n+=1

        return board

    def triangle_board(self, initial_holes):
        '''
        Generate a size*size list with holes.
        Holes are initially filled with pegs.
        Unused parts of board filled with "UNUSED nodes".
        '''
        # Create a empty (zero-filled) size x size matrix, able to contain Holes
        board = np.zeros((self.size, self.size), Node)
        # Fill the board with Nodes.
        for row in range(0,self.size):
            for col1 in range(0, row+1):
                # Adding pegged nodes
                board[row][col1] = Node(Status.PEG)
            for col2 in range(row+1, self.size):
                # Adding unused nodes
                board[row][col2] = Node(Status.UNUSED)
        
        '''Now initialize the holes'''
        # Number of holes inserted. (Initial_holes >= 1)
        n = 1
        # Initial hole created close to the center of the triangle
        board[(self.size//2)][(self.size//2)//2].set_status(Status.EMPTY)

        # Crude, but doing the job
        while n < initial_holes:
            row = rd.randint(0, self.size-1)
            col = rd.randint(0, self.size-1)

            if (board[row][col].get_status() == Status.PEG):
                board[row][col].set_status(Status.EMPTY)
                n+=1

        return board