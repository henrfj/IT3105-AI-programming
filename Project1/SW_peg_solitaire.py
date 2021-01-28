'''
The simulated worlds (SW) for this project, peg solitaire.

'''

from board import *


class SW:
    '''
    SimWorld class for the game peg "solitaire"
    '''
    def __init__(self, board_type: BT, size, k):
        self.board = Board(board_type, size, k)
        self.current_state = self.board.state


    def child_states(self, state):
        # Finds all possible child-states given a parent state
        child_states = {}
        return child_states

    def final_state(self, state):
        # Determines if state is final state
        pass

    def reward(self, state_1, state_2):
        # Returns reward for transition
        pass