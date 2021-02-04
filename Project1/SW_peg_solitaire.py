'''
The simulated worlds (SW) for this project, peg solitaire.

'''

from board import *
import copy

class SW:
    '''
    SimWorld class for the game peg "solitaire"
    '''
    def __init__(self, board_type: BT, size, k):
        #
        self.board = Board(board_type, size, k)

        self.initial_state = self.board.state
        
        self.board_size = size

        self.board_type = board_type

    def make_move(self, state, pos1, pos2):
        ''' 
        Makes a move, and returns the new state
        Moves one peg and removes the skipped peg
        pos1 = hole, pos2 = peg to move. (row, col)
        '''
        
        # Need a deep copy of the array, as
        # in python, arrays are call-by reference always
        new_state = copy.deepcopy(state)

        # Add a peg to the previously empty socket (pos1)
        if new_state[pos1[0],pos1[1]].get_status() != Status.EMPTY:
            raise Exception("This should have been empty")
        new_state[pos1[0],pos1[1]].set_status(Status.PEG)

        # Remove the peg that was skipped over:
        row_distance = pos1[0] - pos2[0]
        col_distance = pos1[1] - pos2[1]
        
        row_pos = int(pos1[0] - row_distance/2)
        col_pos = int(pos1[1] - col_distance/2)

        if new_state[row_pos,col_pos].get_status() != Status.PEG:
            raise Exception("This should have been PEGGED")
        new_state[row_pos,col_pos].set_status(Status.EMPTY)

        # Remove the peg from where it skipped (pos2)
        if new_state[pos2[0],pos2[1]].get_status() != Status.PEG:
            raise Exception("This should have been PEGGED")
        new_state[pos2[0],pos2[1]].set_status(Status.EMPTY)

        return new_state

    def child_states(self, state, board_type, board_size):
        # Finds all possible child-states given a parent state

        # Maybe smart to dictionarize this array
        child_states = []
        for row in range(board_size):
            for col in range(board_size):
                if state[row][col].get_status() == Status.EMPTY:
                # Empty hole found

                    # Direction north
                    if(row-1) >= 0:
                        # In bound
                        if state[row-1][col].get_status() == Status.PEG:
                            # This peg can be skipped
                            if (row-2) >= 0:
                                # In bound
                                if state[row-2][col].get_status() == Status.PEG:
                                    # This peg can jump
                                    child_states.append(self.make_move(state, (row, col), (row-2, col)))

                    # Direction west
                    if(col-1) >= 0:
                        # In bound
                        if state[row][col-1].get_status() == Status.PEG:
                            # This peg can be skipped
                            if (col-2) >= 0:
                                # In bound
                                if state[row][col-2].get_status() == Status.PEG:
                                    # This peg can jump
                                    child_states.append(self.make_move(state, (row, col), (row, col-2)))

                    # Direction south
                    if(row+1)<board_size:
                        # In bound
                        if state[row+1][col].get_status() == Status.PEG:
                            # This peg can be skipped
                            if (row+2) < board_size:
                                # In bound
                                if state[row+2][col].get_status() == Status.PEG:
                                    # This peg can jump
                                    child_states.append(self.make_move(state, (row, col), (row+2, col)))
                
                    # Direction east
                    if(col+1) < board_size:
                        # In bound
                        if state[row][col+1].get_status() == Status.PEG:
                            # This peg can be skipped
                            if (col+2) < board_size:
                                # In bound
                                if state[row][col+2].get_status() == Status.PEG:
                                    # This peg can jump
                                    child_states.append(self.make_move(state, (row, col), (row, col+2)))
                
                    if board_type == BT.DIAMOND: #Diamond board

                        # Direction south-west: (r+1, c-1)
                        if (col-1)>=0 and (row+1) < board_size:
                            # In bound
                            if state[row+1][col-1].get_status() == Status.PEG:
                                # This peg can be skipped
                                if (col-2) >= 0 and (row+2) < board_size:
                                    # In bound
                                    if state[row+2][col-2].get_status() == Status.PEG:
                                        # This peg can jump
                                        child_states.append(self.make_move(state, (row, col), (row+2, col-2)))
                                    
                        # Direction north-east: (r-1, c+1)
                        if (row-1)>=0 and (col+1) < board_size:
                            # In bound
                            if state[row-1][col+1].get_status() == Status.PEG:
                                # This peg can be skipped
                                if (row-2) >= 0 and (col+2) < board_size:
                                    # In bound
                                    if state[row-2][col+2].get_status() == Status.PEG:
                                        # This peg can jump
                                        child_states.append(self.make_move(state, (row, col), (row-2, col+2)))
                
                    else: #Triangle board
                    
                        # Direction south-east: (r+1, c+1)
                        if (col+1) < board_size and (row+1) < board_size:
                            # In bound
                            if state[row+1][col+1].get_status() == Status.PEG:
                                # This peg can be skipped
                                if (col+2) < board_size and (row+2) < board_size:
                                    # In bound
                                    if state[row+2][col+2].get_status() == Status.PEG:
                                        # This peg can jump
                                        child_states.append(self.make_move(state, (row, col), (row+2, col+2)))

                        # Direction north-west: (r-1, c-1)
                        if (row-1) >= 0 and (col-1) >= 0:
                            # In bound
                            if state[row-1][col-1].get_status() == Status.PEG:
                                # This peg can be skipped
                                if (row-2) >= 0 and (col-2) >= 0:
                                    # In bound
                                    if state[row-2][col-2].get_status() == Status.PEG:
                                        # This peg can jump
                                        child_states.append(self.make_move(state, (row, col), (row-2, col-2)))

        return child_states

    def final_state(self, state, board_type, board_size):
        '''Finds out if there are no more possible moves. 
            Returns True if that is the case.
        '''
        moves = self.child_states(state, board_type, board_size)
        # No more moves left
        if len(moves) == 0:
            return True
        return False

    def state_to_array(self, state, size):
        '''Translates a board to a simple np.array
        '''
        arr = np.zeros((size, size), int)
        for row in range(size):
            for col in range(size):
                if state[row][col].get_status() == Status.PEG:
                    arr[row][col] = 1
                elif state[row][col].get_status() == Status.EMPTY:
                    arr[row][col] = 0
                elif state[row][col].get_status() == Status.UNUSED:
                    arr[row][col] = -1
                else: 
                    raise Exception("This is no state, wrong Statuses")
        return arr

    def reward(self, state_1, state_2, board_type, board_size):
        '''
        Returns reward for a transition
        Idea: favourise removing edge-pins, as you need them down first.
        '''
        edge_pin_reward = 50
        winning_reward = 1000
        
        pass