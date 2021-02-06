'''
Test functions. Testing the simworld and the visualization of it.

'''

from SW_peg_solitaire import *
from SW_visualization import Display


def visualization_test():
    '''
    '''
    
    # First, make and print a SW.
    size = 5
    k = 1
    sim = SW(BT.TRIANGLE, size, k)
    print_state(sim.board.state, sim.board_type, sim.board_size)
    
    # Then try to display it using Display
    arr_state = sim.state_to_array(sim.board.state, sim.board_size)

    display = Display()
    display.display_board(arr_state, sim.board_type, sim.board_size)
    
    
    # First, make and print a SW.
    size = 3
    k = 1
    sim = SW(BT.DIAMOND, size, k)
    print_state(sim.board.state, sim.board_type, sim.board_size)
    
    # Then try to display it using Display
    arr_state = sim.state_to_array(sim.board.state, sim.board_size)

    display = Display()
    display.display_board(arr_state, sim.board_type, sim.board_size)


    

def pins_left_test():
    # First, make and print a SW
    size = 7
    k = 8
    sim = SW(BT.DIAMOND, size, k)
    print_state(sim.board.state, sim.board_type, sim.board_size)
    print("PEGs left:", sim.pins_left(sim.board.state, sim.board_type,  sim.board_size))

    # With display!
    display = Display()
    while(not sim.final_state(sim.board.state, sim.board_type, sim.board_size)):
        child_states = sim.child_states(sim.board.state, sim.board_type, sim.board_size)
        sim.set_board_state(child_states[0])
        
        # Print result
        #print_state(sim.board.state, sim.board_type, sim.board_size)
        #print("PEGs left:", sim.pins_left(sim.board.state, sim.board_type,  sim.board_size))

        # Display result
        arr_state = sim.state_to_array(sim.board.state, sim.board_size)
        display.display_board(arr_state, sim.board_type, sim.board_size)

    sim.reward(sim.board.state, sim.board.state, sim.board_type, sim.board_size)

def find_child_states():
    # First, make and print a SW
    size = 7
    k = 1
    sim = SW(BT.DIAMOND, size, k)
    print_state(sim.board.state, sim.board_size, sim.board_type)

    # Then, print out child-states
    child_states = sim.child_states(sim.board.state, sim.board_type, sim.board_size)
    print("Possible child-states")
    for child in child_states:
        print_state(child, sim.board_size, sim.board_type)

def convert_to_matrix():
    # First, make a SW
    size = 7
    k = 1
    sim = SW(BT.TRIANGLE, size, k)
    arr = sim.state_to_array(sim.board.state, sim.board_size)

    print(arr)

def create_diamond_board():
    '''
    Testing the board creation method
    '''
    # size x size  board, with k holes
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
    Function for printing board's states, used for debugging.
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

def print_state(state, type, size):
    '''
    Function for printing states, used for debugging.
    '''
    print("Board type:", str(type))
    s = "["
    for row in range(0, size):
        s+="["
        for col in range(0, size):
            if state[row][col].get_status() == Status.EMPTY:
                s += " 0 "
            elif(state[row][col].get_status() == Status.PEG):
                s += " 1 "
            else:
                # Not used
                s += " - "
        if row!=size-1:
            s += "]\n "
        else:
            s+="]"

    s += "]"
    print(s)