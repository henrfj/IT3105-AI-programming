'''
Module containing the simulated worlds (SW) for this project,
a game of "peg solitaire".
'''

# Libraries
import copy
import random as rd
from enum import Enum
import numpy as np


# Enum  for easier to read code
class BT(Enum):
    DIAMOND = 0
    TRIANGLE = 1

class SW:
    '''
    SimWorld class for the game peg "solitaire"
    '''
    def __init__(self, board_type: BT, board_size, initial_holes):
        # Setting recurring parameters
        self.board_size = board_size
        self.board_type = board_type
        self.initial_holes = initial_holes # List of tuples: locations of initial holes.

        # Initiate some initial board, then update it
        #self.state = np.zeros(board_size**2)
        self.new_board()
        # To have same board over several episodes
        self.base_board = np.copy(self.state)

    # Makes a hypothetical move, returns the resulting state.
    def make_move(self, state, pos1, pos2):
        ''' 
        Makes a single move, and returns the new state.
        Moves one peg and removes the skipped peg. Used to find child states.
        pos1 = hole, pos2 = peg to move. (row, col)
        '''
        
        # Need a deep copy of the array, as
        # in python, arrays are call-by reference always
        new_state = copy.deepcopy(state)

        # Add a peg to the previously empty socket (pos1)
        if new_state[pos1[0],pos1[1]] != 0:
            raise Exception("This should have been empty")
        new_state[pos1[0],pos1[1]] = 1

        # Remove the peg that was skipped over:
        row_distance = pos1[0] - pos2[0]
        col_distance = pos1[1] - pos2[1]
        
        row_pos = int(pos1[0] - row_distance/2)
        col_pos = int(pos1[1] - col_distance/2)

        if new_state[row_pos,col_pos] != 1:
            raise Exception("This should have been PEGGED")
        new_state[row_pos,col_pos] = 0

        # Remove the peg from where it skipped (pos2)
        if new_state[pos2[0],pos2[1]] != 1:
            raise Exception("This should have been PEGGED")
        new_state[pos2[0],pos2[1]] = 0

        return new_state

    # Returns all child-states of a given state.
    def child_states(self, state):
        '''
        Finds all possible child-states given a parent state
        '''
        board_type = self.board_type
        board_size = self.board_size
        # Maybe smart to dictionarize this array
        child_states = []
        for row in range(board_size):
            for col in range(board_size):
                if state[row][col] == 0:
                # Empty hole found

                    # Direction north
                    if(row-1) >= 0:
                        # In bound
                        if state[row-1][col] == 1:
                            # This peg can be skipped
                            if (row-2) >= 0:
                                # In bound
                                if state[row-2][col] == 1:
                                    # This peg can jump
                                    child_states.append(self.make_move(state, (row, col), (row-2, col)))

                    # Direction west
                    if(col-1) >= 0:
                        # In bound
                        if state[row][col-1] == 1:
                            # This peg can be skipped
                            if (col-2) >= 0:
                                # In bound
                                if state[row][col-2] == 1:
                                    # This peg can jump
                                    child_states.append(self.make_move(state, (row, col), (row, col-2)))

                    # Direction south
                    if(row+1)<board_size:
                        # In bound
                        if state[row+1][col] == 1:
                            # This peg can be skipped
                            if (row+2) < board_size:
                                # In bound
                                if state[row+2][col] == 1:
                                    # This peg can jump
                                    child_states.append(self.make_move(state, (row, col), (row+2, col)))
                
                    # Direction east
                    if(col+1) < board_size:
                        # In bound
                        if state[row][col+1] == 1:
                            # This peg can be skipped
                            if (col+2) < board_size:
                                # In bound
                                if state[row][col+2] == 1:
                                    # This peg can jump
                                    child_states.append(self.make_move(state, (row, col), (row, col+2)))
                
                    if board_type == BT.DIAMOND: #Diamond board

                        # Direction south-west: (r+1, c-1)
                        if (col-1)>=0 and (row+1) < board_size:
                            # In bound
                            if state[row+1][col-1] == 1:
                                # This peg can be skipped
                                if (col-2) >= 0 and (row+2) < board_size:
                                    # In bound
                                    if state[row+2][col-2] == 1:
                                        # This peg can jump
                                        child_states.append(self.make_move(state, (row, col), (row+2, col-2)))
                                    
                        # Direction north-east: (r-1, c+1)
                        if (row-1)>=0 and (col+1) < board_size:
                            # In bound
                            if state[row-1][col+1] == 1:
                                # This peg can be skipped
                                if (row-2) >= 0 and (col+2) < board_size:
                                    # In bound
                                    if state[row-2][col+2] == 1:
                                        # This peg can jump
                                        child_states.append(self.make_move(state, (row, col), (row-2, col+2)))
                
                    else: #Triangle board
                    
                        # Direction south-east: (r+1, c+1)
                        if (col+1) < board_size and (row+1) < board_size:
                            # In bound
                            if state[row+1][col+1] == 1:
                                # This peg can be skipped
                                if (col+2) < board_size and (row+2) < board_size:
                                    # In bound
                                    if state[row+2][col+2] == 1:
                                        # This peg can jump
                                        child_states.append(self.make_move(state, (row, col), (row+2, col+2)))

                        # Direction north-west: (r-1, c-1)
                        if (row-1) >= 0 and (col-1) >= 0:
                            # In bound
                            if state[row-1][col-1] == 1:
                                # This peg can be skipped
                                if (row-2) >= 0 and (col-2) >= 0:
                                    # In bound
                                    if state[row-2][col-2] == 1:
                                        # This peg can jump
                                        child_states.append(self.make_move(state, (row, col), (row-2, col-2)))

        return child_states

    # Decides if a state is a final state.
    def final_state(self, state):
        '''
        Finds out if there are no more possible moves. 
        Returns True if that is the case.
        '''
        board_type = self.board_type 
        board_size = self.board_size 

        moves = self.child_states(state)
        # No more moves left
        if len(moves) == 0:
            return True
        return False

    # Returns number of pegs left standing.
    def pegs_left(self, state):
        '''Calculate number of pins left
        '''
        board_size = self.board_size
        # Number of pins
        n = 0
        for row in range(board_size):
            for col in range(board_size):
                if state[row][col] == 1:
                    n+=1

        return n

    # Returns the reward given for a transition.
    def reward(self, state_1, state_2, mode=0):
        '''
        Returns reward for a transition
        Idea: favourise removing edge-pins, as you need them down first.
        Every move is guaranteed to bring down a pin, so no need to reward that.
        ''' 

        reward = None
        n_pegs = self.pegs_left(state_2)
        if mode == 0: # Best performer.
            
            # Each step rewarded: Two reasons why
            #   1: Having to choose between two already visited paths, 
            #       it will choose the longest (the path where it got closest).
            #   2: If there are no solutions to the board, it will coverge 
            #       to the closest possible solution.
            #   
            #   0.1 is small enough, that it will always prioritize new, unexplored patsh anyways.
            
            reward = 0.1
            if n_pegs == 1:
                print("YOU WON!")
                reward = 100000
            elif self.final_state(state_2):  
                # Lost
                reward = -10
        
        elif mode == 1: # Try to scale the 0 mode.
            # Each step
            reward = 1
            if n_pegs == 1:
                print("YOU WON!")
                reward = 10000
            elif self.final_state(state_2): 
                # Lost
                reward = -20

        elif mode == 2: # Small penalty.
            reward = -0.1
            if n_pegs == 1:
                print("YOU WON!")
                reward = 100000

            elif self.final_state(state_2): 
                # Lost
                reward = -10

        elif mode == 3: # The recomended.
            reward = 0
            if n_pegs == 1:
                print("YOU WON!")
                reward = 100000000000
            elif self.final_state(state_2): 
                # Lost
                reward = -100

        return reward
    
    # Sets the current board state.
    def set_board_state(self, new_state): 
            '''
            Simple setter for updating the state of the board.
            '''
            self.state = new_state

    # Creates a new board, based on parameters.
    def new_board(self):
        '''
        Create a new board for a new game.
        '''
        board_type = self.board_type 
        board_size = self.board_size 
        initial_holes = self.initial_holes

        if (board_type is BT.DIAMOND):
            '''
            Generate a size*size list with nodes.
            Nodes are initially filled with pegs
            '''
            # Create a empty (zero-filled) size x size matrix.
            board = np.zeros((board_size, board_size), int)
            # Fill the board with default size, being filled with pegs.
            for row in range(0,board_size):
                for col in range(0, board_size):
                    board[row][col] = 1

            '''Now initialize the holes'''
            # Initial hole created close to the center.
            # Holes need to be along the horisontal axis.
            if len(initial_holes) == 0:
                board[((board_size)//2)][((board_size-1)//2)] = 0

        elif (board_type is BT.TRIANGLE):
            '''
            TRIANGLE
            Generate a size*size list with holes.
            Holes are initially filled with pegs.
            Unused parts of board filled with "UNUSED nodes".
            '''
            # Create a empty (zero-filled) size x size matrix, able to contain Holes
            board = np.zeros((board_size, board_size), int)
            # Fill the board with Pegs.
            for row in range(0,board_size):
                for col1 in range(0, row+1):
                    # Adding pegged nodes
                    board[row][col1] = 1
                for col2 in range(row+1, board_size):
                    # Adding unused nodes
                    board[row][col2] = 2
            
            '''Now initialize the holes'''
            # Initial hole created close to the center of the triangle.
            if len(initial_holes) == 0:
                board[(board_size//2)][(board_size//2)//2] = 0
            
        else:
            raise Exception("This is no board type")
    
        for pos in initial_holes:
            row = pos[0]
            col = pos[1]
            if row >= self.board_size and col >= self.board_size:
                raise IndexError("Trying to place hole outside board")
            if board[row][col] == 1 or board[row][col] == 0:
                board[row][col] = 0
            else: # outside playing field of triangle
                raise IndexError("Trying to place hole outside playing field")
        
        self.state = board

    # Resets the board between each game.
    def reset_board(self):
        '''
        Reset board in between episodes during learning. 
        '''
        self.state = np.copy(self.base_board)

