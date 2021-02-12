'''
Test functions. Testing the simworld and the visualization of it.

'''
# My own libraries
from SW_peg_solitaire import SW, BT
from SW_visualization import Display

# Other libraries
from matplotlib import animation
import matplotlib.pyplot as plt
import networkx as nx

def animation_test_actions():
    '''
    Try to animate the shit 
    '''
    # First, make and print a SW
    size = 5
    k = 7
    sim = SW(BT.DIAMOND, size, k)
    print_state(sim.state, sim.board_type, sim.board_size)

    # Create display!
    display = Display()
    
    # Animation variables
    moves_used = 0
    states = []
    
    # Generate first state
    states.append(sim.state)


    while(not sim.final_state(sim.state)):
        actions = sim.actions()
        # Choose first and best move

        # TODO: Instead of choosing first move, make an actual RL
        sim.set_board_state(actions[0])
        moves_used += 1

        # Add to list of moves
        states.append(actions[0])



    # Build plot
    fig, ax = plt.subplots(figsize=(6,4))

    def update(num):
        '''
        Made to help animator animate the board.
        '''
        ax.clear()
        state = states[num]
        # Produce a neighbourhood matrix
        neighbour_matrix = display.transform(state, sim.board_type, sim.board_size)
        # Use networkX to make a graph based on the neighbourhood matrix
        display.graph = nx.from_numpy_array(neighbour_matrix)
        # Color nodes according to state matrix
        color_map = display.color_nodes(state, sim.board_type, sim.board_size)
        # Try something new: fixed node-positions
        nodepos = display.node_pos(state, sim.board_type, sim.board_size)
        # Draw the current frame
        nx.draw(display.graph, node_color=color_map, with_labels=True,pos=nodepos)

    # Make animation
    ani = animation.FuncAnimation(fig, update, frames=(moves_used+1), interval=1000, repeat=False)
    plt.show()



# def animation_test():
#     '''
#     Try to animate the shit 
#     '''
#     # First, make and print a SW
#     size = 7
#     k = 4
#     sim = SW(BT.TRIANGLE, size, k)
#     print_state(sim.board.state, sim.board_type, sim.board_size)

#     # Create display!
#     display = Display()
    
#     # Animation variables
#     moves_used = 0
#     arr_states = []
    
#     # Generate first arr_state
#     arr_state = sim.state_to_array(sim.board.state, sim.board_size)
#     arr_states.append(arr_state)


#     while(not sim.final_state(sim.board.state, sim.board_type, sim.board_size)):
#         child_states = sim.child_states(sim.board.state, sim.board_type, sim.board_size)
#         # Choose first and best move

#         # TODO: Instead of choosing first move, make an actual RL
#         sim.set_board_state(child_states[0])
#         moves_used += 1

#         # Add to list of moves
#         arr_state = sim.state_to_array(sim.board.state, sim.board_size)
#         arr_states.append(arr_state)



#     # Build plot
#     fig, ax = plt.subplots(figsize=(6,4))

#     def update(num):
#         '''
#         Made to help animator animate the board.
#         '''
#         ax.clear()
#         state = arr_states[num]
#         # Produce a neighbourhood matrix
#         neighbour_matrix = display.transform(state, sim.board_type, sim.board_size)
#         # Use networkX to make a graph based on the neighbourhood matrix
#         display.graph = nx.from_numpy_array(neighbour_matrix)
#         # Color nodes according to state matrix
#         color_map = display.color_nodes(state, sim.board_type, sim.board_size)
#         # Try something new: fixed node-positions
#         nodepos = display.node_pos(state, sim.board_type, sim.board_size)
#         # Draw the current frame
#         nx.draw(display.graph, node_color=color_map, with_labels=True,pos=nodepos)

#     # Make animation
#     ani = animation.FuncAnimation(fig, update, frames=(moves_used+1), interval=1000, repeat=False)
#     plt.show()

#     sim.reward(sim.board.state, sim.board.state, sim.board_type, sim.board_size)

# def visualization_test():
#     '''
#     '''
    
#     # First, make and print a SW.
#     size = 5
#     k = 9
#     sim = SW(BT.TRIANGLE, size, k)
#     print_state(sim.board.state, sim.board_type, sim.board_size)
    
#     # Then try to display it using Display
#     arr_state = sim.state_to_array(sim.board.state, sim.board_size)

#     display = Display()
#     display.display_board(arr_state, sim.board_type, sim.board_size)
    
    
#     # First, make and print a SW.
#     size = 3
#     k = 1
#     sim = SW(BT.DIAMOND, size, k)
#     print_state(sim.board.state, sim.board_type, sim.board_size)
    
#     # Then try to display it using Display
#     arr_state = sim.state_to_array(sim.board.state, sim.board_size)

#     display = Display()
#     display.display_board(arr_state, sim.board_type, sim.board_size)

# def pins_left_test():
#     # First, make and print a SW
#     size = 6
#     k = 3
#     sim = SW(BT.TRIANGLE, size, k)
#     print_state(sim.board.state, sim.board_type, sim.board_size)
#     print("PEGs left:", sim.pins_left(sim.board.state, sim.board_type,  sim.board_size))

#     # With display!
#     display = Display()

#     # Display result
#     arr_state = sim.state_to_array(sim.board.state, sim.board_size)
#     display.display_board(arr_state, sim.board_type, sim.board_size)


    
#     while(not sim.final_state(sim.board.state, sim.board_type, sim.board_size)):
#         child_states = sim.child_states(sim.board.state, sim.board_type, sim.board_size)
#         sim.set_board_state(child_states[0])
        
#         # Print result
#         print_state(sim.board.state, sim.board_type, sim.board_size)
#         print("PEGs left:", sim.pins_left(sim.board.state, sim.board_type,  sim.board_size))

#         # Display result
#         arr_state = sim.state_to_array(sim.board.state, sim.board_size)
#         display.display_board(arr_state, sim.board_type, sim.board_size)

#     sim.reward(sim.board.state, sim.board.state, sim.board_type, sim.board_size)

# def find_child_states():
#     # First, make and print a SW
#     size = 7
#     k = 1
#     sim = SW(BT.DIAMOND, size, k)
#     print_state(sim.board.state, sim.board_size, sim.board_type)

#     # Then, print out child-states
#     child_states = sim.child_states(sim.board.state, sim.board_type, sim.board_size)
#     print("Possible child-states")
#     for child in child_states:
#         print_state(child, sim.board_size, sim.board_type)

# def convert_to_matrix():
#     # First, make a SW
#     size = 7
#     k = 1
#     sim = SW(BT.TRIANGLE, size, k)
#     arr = sim.state_to_array(sim.board.state, sim.board_size)

#     print(arr)

# def create_diamond_board():
#     '''
#     Testing the board creation method
#     '''
#     # size x size  board, with k holes
#     size = 5
#     k = 1
#     board = Board(BT.DIAMOND, size, k)
#     printboard(board, size)
    
# def create_triangle_board():
#     '''
#     Testing the board creation method for triangle board.
#     '''
#     size = 5 # Height of triangle
#     k = 1
#     board = Board(BT.TRIANGLE, size, k)
#     printboard(board, size)

# def printboard(board, size):
#     '''
#     Function for printing board's states, used for debugging.
#     '''
#     print("Board:", str(board.type))
#     s = "["
#     for row in range(0, size):
#         s+="["
#         for col in range(0, size):
#             if board.state[row][col].get_status() == Status.EMPTY:
#                 s += " 0 "
#             elif(board.state[row][col].get_status() == Status.PEG):
#                 s += " 1 "
#             else:
#                 # Board not used
#                 s += " - "
#         if row!=size-1:
#             s += "]\n "
#         else:
#             s+="]"

#     s += "]"
#     print(s)

def print_state(state, type, size):
    '''
    Function for printing states, used for debugging.
    '''
    print("Board type:", str(type))
    s = "["
    for row in range(0, size):
        s+="["
        for col in range(0, size):
            if (state[row][col] == 0):
                s += " O "
            elif(state[row][col] == 1):
                s += " P "
            else:
                # Not used
                s += " - "
        if row != (size-1):
            s += "]\n "
        else:
            s+="]"
    s += "]"
    print(s)
