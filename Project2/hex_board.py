import numpy as np




class hex_board:
    '''
    Hex board. Game logic. State manager.
    '''

    def __init__(self, k):
        '''Init'''
        self.k = k
        self.initialize_states()
        self.player_turn = 0       # ID 0 (p1) and 1 (p2) for whos turn it is.
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
        
        # Contains (x,x) - connection to the two edges
        value = np.empty((), dtype=object)
        value[()] = (0, 0)
        self.edge_connections = np.full((k, k), value, dtype=object)
         

    def flatten_state(self, state):
        ''' Flatten state to 1D array. Add player ID tag at beginning.'''
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
        children = {}
        for i in range(len(self.possible_moves)):
            if self.possible_moves[i] == 1: # Spot is free.
                child = np.copy(self.state)
                col = i%self.k
                row = i//self.k
                child[row][col] = self.player_turn+1 # Player turn is 0 indexed.
                children[child] = (row,col) # Position of last move. Used to display.
        return children
        
    def make_move(self, new_state, pos):
        '''Make a move from current state, to new_state'''
        # Make sure to make a copy
        self.state = np.copy(new_state)
        
        # Update the idle state
        # TODO: pos is converted from 1D -> 2D -> 1D within a single move. Is it needed?
        # Could keep it as 1D. But 2D is easier for displaying i believe.
        row = pos[0]
        col = pos[1]
        if self.possible_moves[row*self.k + col] != 1:
            raise Exception("Try to make illegal move.") 
        else:
            self.possible_moves[row*self.k + col] = 1
        
        # Update the edge-states.
        # 1
        if self.on_edge(self.player_turn+1, pos):
            self.spread_news(self.player_turn+1, pos)
        # 2
        self.copy_neighbours(self.player_turn+1, pos)

        # Check for victory
        # TODO: should check for victory each time we update the edge_connection

        # change player ID
        if self.player_turn == 0:
            self.player_turn = 1
        else:
            self.player_turn = 0


    def on_edge(self, player_ID, pos):
        '''Checks to see if the position is an edge position for the given player
        Player ID: {1,2}
        Pos: (row,col)
        '''
        pass

    def get_neigbours(self, player_ID, pos):
        '''Get position of all friendly neighbours
        Player ID: {1,2}
        Pos: (row,col)'''
        pass

    def copy_neighbours(self, player_ID, pos):
        '''Copy the data of all friendly neighbours
        Player ID: {1,2}
        Pos: (row,col)'''
        pass

    def spread_news(self, player_ID, pos):
        ''' Recursively updates neighbour data until the news are old
        Player ID: {1,2}
        Pos: (row,col)'''
        pass
    
    def final_state(self, pos):
        ''' Cheks for V by looking for a (1,1) in the edge_connection
        Player ID: {1,2}
        Pos: (row,col)'''
        pass




