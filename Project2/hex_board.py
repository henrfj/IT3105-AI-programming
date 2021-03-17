# IMPORTS
import numpy as np # States kept as np arrays.

class hex_board:
    '''
    Hex board. Game logic. State manager.
    '''

    def __init__(self, k):
        '''Init'''
        self.k = k
        self.initialize_states()
        self.player_turn = 1       # ID 1 (p1) and 2 (p2) for whos turn it is.
        self.initialize_states()

        # Game over
        self.game_over = False
        self.winner = 0

    def initialize_states(self):
        '''Initialize all states. Can be used to reset game.'''
        k = self.k
        
        # 2D array. Contains 0s, 1s and 2s. Player 1, 2 and empty.
        self.state = np.zeros((k,k))            
        
        # Contains 0s and 1s. 1D to find moves.
        self.possible_moves = np.ones(k*k)     
        
        # Contains (x,x) - connection to the two edges. 3D array.
        self.edge_connections = np.zeros((k, k, 2))
        
    def flatten_state(self, state):
        ''' Flatten state to 1D array. Add player ID tag at beginning.'''
        # TODO: look into the (0,0), (1,0), (0,1) representation. 
        k = self.k
        flat = np.zeros(k**2+1)
        for row in range(k):
            for col in range(k):
                flat[row*k+col+1] = self.state[row][col]
        # Add ID tag.
        flat[0] = self.player_turn
        return flat

    def child_states(self):
        '''Returns list of all child states given current state.'''
        # Contains list of child-states, and pos. of newly placed piece.
        children = []
        for i in range(len(self.possible_moves)):
            if self.possible_moves[i] == 1: # Spot is free.
                child = np.copy(self.state)
                col = i%self.k
                row = i//self.k
                child[row][col] = self.player_turn
                children.append([child, (row,col)]) # [New_state, last move (row, col)]
        return children
        
    def make_move(self, new_state, pos):
        '''Make a move from current state, to new_state'''        
        # Make sure to make a copy
        self.state = np.copy(new_state)
        
        # Remove 
        row = pos[0]
        col = pos[1]
        if self.possible_moves[row*self.k + col] != 1:
            raise Exception("Trying to make illegal move!") 
        else:
            self.possible_moves[row*self.k + col] = 0 
        
        # Set initial edge-config
        self.on_edge(self.player_turn, pos)
        
        # Spread news around to neighbours, if any.
        self.spread_news(self.player_turn, pos)
        
        # Check for victory
        self.final_state(self.player_turn, pos)

        # change player turn / player ID
        if self.player_turn == 1:
            self.player_turn = 2
        else:
            self.player_turn = 1

    def on_edge(self, player_ID, pos):
        '''Checks to see if the position is an edge position for the given player
        Sets the corner ID of self.edge_connections (1,0), (0,1) or (0,0)
        Player ID: {1,2}
        Pos: (row,col)
        '''
        # Translate pos to row / col.
        row = pos[0]
        col = pos[1]

        # Player 1 has the top and bottom row. state[0][:] and state [k-1][:]
        if player_ID == 1:
            if row == 0: # TOP
                self.edge_connections[row][col] = (1,0)
            elif row == self.k-1: # Bottom
                self.edge_connections[row][col] = (0,1)
            else: # No edge
                self.edge_connections[row][col] = (0,0) # No edge
        # Player 2 has the left and right edges. State[:][0] and state [:][k-1]
        elif player_ID == 2:
            if col == 0: # Left
                self.edge_connections[row][col] = (1,0)
            elif col == self.k-1: # Right
                self.edge_connections[row][col] = (0,1)
            else: # No edge
                self.edge_connections[row][col] = (0,0) 
        else:
            raise Exception("Invalid player ID in on-edge check.")

    def get_neigbours(self, player_ID, pos):
        '''Get position of all friendly neighbours, simple diamond-logic used.
        Player ID: {1,2}
        Pos: (row,col)'''
        row = pos[0]
        col = pos[1]
        neighbours_pos = [] # filled with pos := (row,col)
        # DOWN
        if (row-1) >= 0:
            if self.state[row-1][col] == player_ID:
                neighbours_pos.append((row-1, col))
        # UP
        if (row+1) <= (self.k-1):    
            if self.state[row+1][col] == player_ID:
                neighbours_pos.append((row+1, col))
        # LEFT
        if (col-1) >= 0:
            if self.state[row][col-1] == player_ID:
                neighbours_pos.append((row, col-1))
        # RIGHT
        if (col+1) <= (self.k-1):
            if self.state[row][col+1] == player_ID:
                neighbours_pos.append((row, col+1))
        # NORTH EAST
        if (col+1) <= (self.k-1) and (row-1) >=0:
            if self.state[row-1][col+1] == player_ID:
                neighbours_pos.append((row-1, col+1))
        # SOUTH WEST
        if (col-1) >= 0 and (row+1) <= (self.k-1):
            if self.state[row+1][col-1] == player_ID:
                neighbours_pos.append((row+1, col-1))
        # Return the list of all neighbour pos on the board.
        return neighbours_pos

    def spread_news(self, player_ID, pos):
        '''Copy the data to/from all friendly neighbours recursively
        Player ID: {1,2}
        Pos: (row,col)
        # TODO: Now we ofte check neighbours more than once.
        '''
        row = pos[0]
        col = pos[1]
        
        neighbours_pos = self.get_neigbours(player_ID, pos)
        got_update = False  # Do we need to spread any news?
        gave_update = False

        for n_pos in neighbours_pos:
            n_row = n_pos[0]
            n_col = n_pos[1]
            
            # From neighbours
            # Connection to one edge
            if self.edge_connections[row][col][0] == 0 and self.edge_connections[n_row][n_col][0] == 1:
                self.edge_connections[row][col][0] = 1
                got_update = True
            # Connection to the other.
            if self.edge_connections[row][col][1] == 0 and self.edge_connections[n_row][n_col][1] == 1:
                self.edge_connections[row][col][1] = 1
                got_update = True
            
            # To neighbours
            # Connection to one edge
            if self.edge_connections[n_row][n_col][0] == 0 and self.edge_connections[row][col][0] == 1:
                self.edge_connections[n_row][n_col][0] = 1
                gave_update = True
            # Connection to the other.
            if self.edge_connections[n_row][n_col][1] == 0 and self.edge_connections[row][col][1] == 1:
                self.edge_connections[n_row][n_col][1] = 1
                gave_update = True

            # Keep spreading
            # TODO: Problem! What if the last neighbour is the one that has the news.
            if got_update:
                self.spread_news(player_ID, pos) # Restart spreading to every single neighbour.
                break # We restarted spreading, no need to do it again.
            if gave_update: 
                self.spread_news(player_ID, n_pos) # Spread around the reciever.
                gave_update = False

    def final_state(self, player_ID, pos):
        ''' Cheks for V by looking for a (1,1) in the edge_connection
        Player ID: {1,2}
        Pos: (row,col)'''
        if self.edge_connections[pos[0]][pos[1]][0] == 1 and self.edge_connections[pos[0]][pos[1]][1] == 1:
            #print("Player",player_ID,"won!")
            self.game_over = True
            self.winner = player_ID



