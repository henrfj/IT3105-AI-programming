# IMPORTS
import matplotlib.pyplot as plt # Plotting
import networkx as nx # Network graph
from matplotlib import animation # Animating network graphs
import numpy as np # Efficient arrays

### TESTING - ROTATION?
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes


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

        # Update function to update frames of animation.
        # 
        nodepos = self.node_pos(board_size) # TODO: only need to do this once!!
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

    def node_pos(self, board_size):
        nodepos={} # Nodepos needs to be a dict.
        array_2d = np.ndarray((board_size, board_size), dtype=tuple)
        # 1
        # Fill 2d array with initial, un-rotated coordinates of the nodes. 
        for i in range(board_size*board_size):
            row = i // board_size
            col = i - row*board_size
            array_2d[row, col] = (col, row) # The x and y are reversed.
            #nodepos[i] = (col, row)
        
        # 2
        # Rotate coordinates clockwise 45 degree.
        # Then fill nodepos with these rotated coordinates. 
        for i in range(board_size):
            for j in range(board_size):
                (x,y) = array_2d[i,j]
                array_2d[i,j] = (x-y, y+x) # Found using complex number algebra!
                nodepos[i*board_size+j] = array_2d[i,j]

        return nodepos
