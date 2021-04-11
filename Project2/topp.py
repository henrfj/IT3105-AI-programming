'''Tournament of progressive policies (TOPP)'''


# Usefull modules.
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
# Own modules.
from hex_board import hex_board
from actor import TOPP_Actor
from hex_display import hex_display

class TOPP:
    '''The tournament manager, pitting progressive policies against each other and displaying results.'''
    def __init__(self, k : int, frame_delay=500, figsize=(15,15)):
        '''Initializes the tournament handler.'''
        self.k = k # Board size that agents/actors have trained on.
        self.display = hex_display(frame_delay, figsize) # For visualizing single games.
    #
    def stochastic_round_robin(self, agent_paths, no_games : int, interval=20, visualize_topp_games=False, agent_to_watch=[]):
        ''' Simulates a tournament between agents, choosing moves stochastically based on their distributions. 
        IN: - Agent_paths is a list of paths to the agents.
            - no_games is the games each player will play against each other in a series.
        OUT: prints out some results, and might display some too.
        '''
        no_agents = len(agent_paths)
        winnings = np.zeros((no_agents, no_agents)) # A map of all winnings.
        print("============================================")
        print("===== TOPP between",no_agents,"different agents! =====")
        print("============================================")
        
        # Play the tournament
        for i in range(no_agents):
            model1 = keras.models.load_model(agent_paths[i])
            actor1 = TOPP_Actor(model1)
            for j in range(i+1, no_agents):
                # Every player plays against all following agents in a series.
                print("Tournament in progress ...")
                model2 = keras.models.load_model(agent_paths[j])
                actor2 = TOPP_Actor(model2)
                p1_w = self.stochastic_single_series(actor1, actor2, no_games)
                winnings[i][j] = p1_w
                winnings[j][i] = no_games - p1_w
                if visualize_topp_games:
                    for k in range(len(agent_to_watch)):
                        if i in agent_to_watch[k] and j in agent_to_watch[k]:
                            print("============================================")
                            print("A FIGHT BETWEEN AGENT",i,"(as player 1, red) and",j,"(as player 2, black)!")
                            print("============================================")
                            self.display_a_match(actor1, actor2)
                            agent_to_watch[k] = (-1, -1) # Remove from list.
                            
                
        # Plot the winnings of each actor
        print("WINNINGS:\n",winnings)
        
        # Counting up the wins.
        scores = np.zeros(no_agents) # Index is the agent
        t = np.linspace(0,no_agents-1, no_agents) * interval # For plotting
        for i in range(no_agents):
            for j in range(no_agents):
                scores[i] += winnings[i][j]

        # Get winrates.
        #scores = (scores - scores.min()) / (scores.max() - scores.min())
        # Smooth out the rates
        coef = np.polyfit(t,scores,3)
        poly1d_fn = np.poly1d(coef)
        # Simple plot of the wins.
        print("SCORES:", scores)
        print("T:", t)
        plt.plot(t, scores, "ob")       # Points
        plt.plot(t, poly1d_fn(t), "-g") # Smooth graph
        plt.title("No. won rounds VS No. Episodes trained.")
        plt.xlabel("Training progress, no. episodes.")
        plt.ylabel("No. won series in the TOPP.")
        plt.show()
    #
    def stochastic_single_series(self, actor1, actor2, no_games : int):
        '''Plays a single series between two actors, and returns statistics.'''
        # The count of p1 wins. 
        p1_w = 0
        # Requies a board to be played on.
        board = hex_board(self.k, verbose=False)
        # Model 1 starts the first game.
        start_player = 1
        # Run through the number of games.
        z = np.random.uniform(0,1, no_games)
        for i in range(no_games):
            # Reset the board. New game.
            board.initialize_states() 
            player_turn = start_player
            # Play one game
            while board.game_over == False:
                # The player whos turn it is takes makes a move.
                if z[i] > 0.1: # Make greedy move.
                    if player_turn==1:
                        move_dist = actor1.move_distribution(board.flatten_state())
                    else:
                        move_dist = actor2.move_distribution(board.flatten_state())
                    # Normalize the distribution.
                    legal_moves =  np.multiply(move_dist, board.possible_moves) # Remove impossible moves.
                    norm_moves = legal_moves / np.sum(legal_moves)

                    # Stochastic choice based on the distribution.
                    #index = np.random.choice(np.arange(0,self.k**2,1), p=norm_moves) # Index of the best move, given a 1D board.
                    index = np.argmax(norm_moves)
                    row = index//self.k
                    col = index%self.k
                    move = (row, col)
                else: # Make a random move.
                    move = board.random_move()

                # Make the move
                board.make_move(move)

                # Change player turn.
                if player_turn == 1:
                    player_turn = 2
                else:
                    player_turn = 1

            # Sees who won.
            if player_turn == 2: # player 1 made the winning move.
                p1_w += 1
        
            # Change who starts for next turn.
            if start_player == 1:
                start_player = 2
            else:
                start_player = 1

            
        return p1_w
    #
    def display_a_match(self, actor1, actor2):
        # Board to play on
        board = hex_board(self.k, verbose=False)
        # Record episodes for displaying.
        episode = [board.state]
        while board.game_over == False:
                # The player whos turn it is takes makes a move.
                if board.player_turn==1:
                    move_dist = actor1.move_distribution(board.flatten_state())
                else:
                    move_dist = actor2.move_distribution(board.flatten_state())
                # Normalize the distribution.
                legal_moves =  np.multiply(move_dist, board.possible_moves) # Remove impossible moves.
                norm_moves = legal_moves / np.sum(legal_moves)

                # Stochastic choice based on the distribution.
                index = np.argmax(norm_moves)
                row = index//self.k
                col = index%self.k
                move = (row, col)

                # Make the move
                board.make_move(move)
                episode.append(board.state)

        # Print winner!
        if board.winner==1:
            winner = 1
            c = "red"
        else:
            winner = 2
            c = "black"
        print("=====> And the winner is player", winner,"! (the",c,"player)")
        # Game over, display the game:
        self.display.animate_episode(episode, self.k)