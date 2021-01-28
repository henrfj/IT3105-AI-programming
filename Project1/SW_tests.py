'''
Test functions. Testing the simworld and the visualization of it.

'''

from SW_peg_solitaire import *



def create_diamond_board():
    '''
    Testing the board creation method
    '''
    # sizexsize  board, with k holes
    size = 4
    k = 3
    board = Board(BT.DIAMOND, size, k)
    printboard(board, size)
    


def printboard(board, size):
    '''
    Function for printing board states, used for debugging.
    '''
    s = "["
    for row in range(0, size):
        s+="["
        for col in range(0, size):
            if board.state[row][col].get_status() == Status.EMPTY:
                s += " 0 "
            else:
                s +=" 1 "
        if row!=size-1:
            s += "]\n "
        else:
            s+="]"

    s += "]"
    print(s)