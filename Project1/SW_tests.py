'''
Test functions. Testing the simworld and the visualization of it.

'''

from SW_peg_solitaire import *



def create_diamond_board():
    '''
    Testing the board creation method
    '''
    # sizexsize  board, with k holes
    size = 5
    k = 1
    board = Board(BT.DIAMOND, size, k)
    printboard(board, size)
    
def create_triangle_board():
    '''
    Testing the board creation method for triangle board.
    '''
    size = 5 # Height of triangle
    k = 1
    board = Board(BT.TRIANGLE, size, k)
    printboard(board, size)

def printboard(board, size):
    '''
    Function for printing board states, used for debugging.
    '''
    print("Board:", str(board.type))
    s = "["
    for row in range(0, size):
        s+="["
        for col in range(0, size):
            if board.state[row][col].get_status() == Status.EMPTY:
                s += " 0 "
            elif(board.state[row][col].get_status() == Status.PEG):
                s += " 1 "
            else:
                # Board not used
                s += " - "
        if row!=size-1:
            s += "]\n "
        else:
            s+="]"

    s += "]"
    print(s)