# IMPORTS
import matplotlib.pyplot as plt # Plotting
import networkx as nx # Network graph
from matplotlib import animation # Animating network graphs
import numpy as np # Efficient arrays




class hex_display:
    '''Class containing methods used for displatying the hex board'''
    def __init__(self, frame_delay, figsize):
        self.frame_delay = frame_delay
        self.figsize = figsize # Tuple (x,y)

    def animate_episode(self, episode, board_size):

        # Animation variables
        states = episode
        # Build plot
        fig, ax = plt.subplots(figsize=self.figsize)



        def update(num):
            '''
            Made to help animator animate the board.
            '''
            # Clear the board.
            ax.clear()
            # Invert y-axis, as draw uses opposite y than what I intended.
            fig.gca().invert_yaxis()
            # Extract single state.
            state = states[num]
            # Produce a neighbourhood matrix
            neighbour_matrix = self.transform(state, board_size)
            # Use networkX to make a graph based on the neighbourhood matrix
            self.graph = nx.from_numpy_array(neighbour_matrix)
            # Color nodes according to state matrix
            color_map = self.color_nodes(state, board_size)
            # Try something new: fixed node-positions
            nodepos = self.node_pos(state, board_size)
            # Draw the current frame
            nx.draw(self.graph, node_color=color_map, with_labels=True,pos=nodepos)

        # Make animation
        ani = animation.FuncAnimation(fig, update, frames=(len(states)), interval=self.frame_delay, repeat=False)
        plt.show()

    def transform(self, state, board_size):
        ''' Transform state into adjacency matrix'''
        
        adjacency = np.zeros((board_size*board_size, board_size*board_size), int)
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
 
        return adjacency

    def color_nodes(self, state, board_size): 
        '''
        Colors the nodes accordingly.
        p1: red, p2: black, empty: white.
        '''
        color_map = []
        # Iterator
        for i in range(board_size*board_size):
            row = i // board_size
            col = i - row*board_size
            if state[row][col] == 1:
                # Its p1 piece
                color_map.append("red")
            elif state[row][col] == 2:
                # Its p2 piece
                color_map.append("black")
            elif state[row][col] == 0:
                # Its empty
                color_map.append("white")
        return color_map

    def node_pos(self, state, board_size):
        nodepos={}
        #mesh_size = 2*board_size - 1
        #i = 0 # Current node 
        #c = 1 # nodes per layer
        #for row in range(mesh_size): # Rows
        #    if row % 2 == 0: # even row
        #        nodepos[i] = ()
        #    else:
        #        pass
        #
        #    i += 1
        for i in range(board_size*board_size):
            row = i // board_size
            col = i - row*board_size
            nodepos[i] = (col, row)
        return nodepos

