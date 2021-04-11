
# Others modules
import matplotlib.pyplot as plt # Plotting
import networkx as nx # Network graph
from matplotlib import animation # Animating network graphs
import numpy as np # Efficient arrays
from copy import deepcopy # Deep copies
import time # Taking time, check performance.
import tensorflow.keras as ker # NN library

# My own modules
import hex_board as hb
import hex_display as hd
import mcts as MC
import actor as Actor
from RL_agent import Agent
from topp import TOPP

#########################################
############### DEMO RUNS ###############
#########################################

# Parameter demo. Show how the system works under several pivotal parameters.
def pivotal_parameters_demo():
    '''Generate agent -> train -> Run small TOP + Showcase games (display)'''
    ##################################################################################################
    ##################################################################################################
    ##################################################################################################
    ## NN MCTS TRAINING PARAMETERS
    learning_rate = 0.02                # Learning rate of NN
    activation = "tanh"                 # Activation function
    optimizer = ker.optimizers.SGD      # Optimizer used 
    k = 4                               # Size of board
    layers = [50, 50, 50]               # Structure of NN
    eps = 1                             # How random the rollouts are. Decay over the training.
    sim_time = 1                        # How accurate the training data is.
    RBUF_size = 512                     # How much training data we can have.
    mbs = 50                            # How much data we can train on at once, after each actual game.
    actual_games = 40                   # More => more training data.
    epochs = 10                         # How much fitting we want, beware of the overfit.
    ## TOPP PARAMS
    save_model = True                   # Save away models for TOPP
    topp_models = 5                     # No. Topp models stored away during training.
    interval = actual_games/topp_models # Saving away agents for TOPP
    topp_games = 100                    # Games played in TOP
    ## DISPLAY GAME PARAMETERS
    frame_delay = 500                   # ms frame delay
    figsize = (15, 15)                  # Dimensions of animation
    visualize_topp_games = True         # Wether or not we want to visualize topp gameplay
    agent_to_watch = [(0,5), (2,3)]     # The IDs of players we want to see fight! ID is 0-->#topp_models
    visualize_training = True           # Wether or not we want to visualize training gameplay
    episodes_to_watch = [0, 18]         # The no. of the training episodes we want to see.
    ##################################################################################################
    ##################################################################################################
    ##################################################################################################
    # Create the agent, mode 2 (NN actor)
    agent = Agent(learning_rate, layers, k, eps, sim_time, RBUF_size, mbs, 2, activation=activation, optimizer=optimizer, frame_delay=frame_delay, figsize=figsize)
    # Train the agents actor, 
    agent_paths = agent.NN_training_sim(actual_games, interval, epochs, verbose=True, save_model=save_model, visualize_training=visualize_training, episodes_to_watch=episodes_to_watch) 
    # Play in the TOP tournament
    topp = TOPP(k, frame_delay=frame_delay, figsize=figsize)
    topp.stochastic_round_robin(agent_paths, topp_games, interval=interval, visualize_topp_games=visualize_topp_games, agent_to_watch=agent_to_watch)

# Demo the training -> saving models -> Topp
def Complete_demo_of_system():
    '''Generate agent -> train -> Run small TOP + Showcase games (display)'''
    learning_rate = 0.02                # Learning rate of NN
    k = 4                               # Size of board
    layers = [50, 50, 50]               # Structure of NN
    eps = 1                             # How random the rollouts are. Decay over the training.
    sim_time = 1                        # How accurate the training data is.
    RBUF_size = 512                     # How much training data we can have.
    mbs = 50                            # How much data we can train on at once, after each actual game.
    actual_games = 40                   # More => more training data.
    epochs = 10                         # How much fitting we want, beware of the overfit.
    topp_models = 5                     # No. Topp models stored away during training.
    interval = actual_games/topp_models # Saving away agents for TOPP
    activation = "tanh"                 # Activation function
    G = 100                             # Games played in TOP

    # Create the agent, mode 2 (NN actor)
    agent = Agent(learning_rate, layers, k, eps, sim_time, RBUF_size, mbs, 2, activation=activation)
    # Train the agents actor, 
    agent_paths = agent.NN_training_sim(actual_games, interval, epochs, verbose=True, save_model=True) 
    print("Agent is trained!") 
    # Play in the TOP tournament
    topp = TOPP(k)
    topp.stochastic_round_robin(agent_paths, G, interval=interval)

# Demo the TOPP system on pre-trained models.
def TOPP_agent_demo():
    ## INPUT PARAMETERS
    k = 5                               # Size of board being played in tournament.
    games = 100                         # No. games played between each player of the tounrmanet.
    frame_delay = 500                   # ms per frame
    figsize = (15,15)                   # Dimensions of display figure
    visualize_topp_games = True         # Visualize or not
    agent_to_watch=[(0,5), (2,3)]       # What agents we want to see fight it out.

    # Paths to agents participating.
    agent_paths = ["./test_model_0.0","./test_model_20.0","./test_model_40.0","./test_model_60.0","./test_model_80.0","./test_model_100.0",] # Interval = 40
    #agent_paths = ["./topp_model_0.0", "./topp_model_20.0", "./topp_model_40.0", "./topp_model_60.0", "./topp_model_80.0", "./topp_model_100.0"] # Interval = 10
    
    # Run the tournament!
    topp = TOPP(k, frame_delay=frame_delay, figsize=figsize)
    topp.stochastic_round_robin(agent_paths, games, interval=40, visualize_topp_games=visualize_topp_games, agent_to_watch=agent_to_watch)

##########################################################
###################### DEMO METHODS ######################
##########################################################

pivotal_parameters_demo()
#Complete_demo_of_system()
#TOPP_agent_demo()

#########################################################
###################### OTHER TESTS ######################
#########################################################

############################################
############### System tests ###############
############################################
#
def new_MCTS_single_sims():
    k = 3
    board = hb.hex_board(k)
    actor = Actor.Random_actor(k)
    mcts = MC.MCTS(0.2, board, actor, 1)
    print("Running 1 single sims:")
    for i in range(1):
        mcts.single_simulation(i, 0.4)
    #print("Number of nodes in the tree:", len(mcts.N.keys()))
    #print(" Standing on root, Q values and N_v values:")
    #legal = board.possible_moves_pos()
    #print("Q:", end="  ")
    #for a in legal:
    #    print(mcts.Q[(hash(board), a)], end=", ")
    #print("N_v:", end="  ")
    #for a in legal:
    #    print(mcts.N_v[(hash(board), a)], end=", ")
    #print("Running 1000 sims, and looking for any updates.")
    #for i in range(1000):
    #    mcts.single_simulation(i, 0.4)
    #print("Number of nodes in the tree:", len(mcts.N.keys()))
    #print(" Standing on root, Q values and N_v values:")
    #legal = board.possible_moves_pos()
    #print("Q:", end="  ")
    #for a in legal:
    #    print(mcts.Q[(hash(board), a)], end=", ")
    #print("N_v:", end="  ")
    #for a in legal:
    #    print(mcts.N_v[(hash(board), a)], end=", ")
# 
def new_MCTS_one_actual_game():
    # Setup.
    k = 6 # board dimensions
    s = 4 # time of each simulation, in seconds
    frame_delay = 500
    figsize = (8,8)
    # Instances used.
    actor = Actor.Random_actor(k)
    board = hb.hex_board(k) # Actual game board
    mcts = MC.MCTS(0.1, board, actor)
    display = hd.hex_display(frame_delay, figsize)

    # Animation sequence
    episode = [board.state]

    # Run simulations
    mcts.simulate_timed(s, progress=0, verbose=True) 
    possible_moves = board.possible_moves_pos()

    while not board.game_over:
        m = 0
        best_index = 0
        for i in range(len(possible_moves)):
            score = mcts.N_v[(hash(board), possible_moves[i])]
            if score > m:
                m = score
                best_index = i
        
        # Apply "best" move
        best_move = possible_moves[best_index]
        board.make_move(best_move)
        episode.append(board.state)

        # Prune MCTS tree so that root = board state
        mcts.prune_search_tree(board)
        if not board.game_over:
            # Run a new sim
            mcts.simulate_timed(s, progress=0, verbose=True)
            possible_moves = board.possible_moves_pos()
# Can my MCTS approximate a optimal strategy after a given number of simulations?
def MCTS_optimality_test(player_ID):
    # Setup.
    k = 6 # board dimensions.
    s = 2 # time of each simulation, in seconds.
    games = 20 # Games played between the two.
    # Instances used.
    actor = Actor.Random_actor(k)
    board = hb.hex_board(k) # Actual game board
    mcts = MC.MCTS(0.1, board, actor)
    # Winner counter
    mc_player_ws = 0
    # Run the games
    for i in range(games):
        print("========================\nGame number:", i,", Player",player_ID,"has won:",mc_player_ws,"\n========================")
        # Reset mcts, all nodes removed.
        board.initialize_states()
        mcts.initialize(board)
        # Run the game
        while not board.game_over:
            # Run simulations, for both turns.
            mcts.simulate_timed(s, progress=0, verbose=True) # run hundreds of sims
            possible_moves = board.possible_moves_pos()
            # player player_ID
            if board.player_turn==player_ID:
                m = 0
                best_index = 0
                for i in range(len(possible_moves)):
                    score = mcts.N_v[(hash(board), possible_moves[i])]
                    if score > m:
                        m = score
                        best_index = i
                
                # Apply "best" move
                best_move = possible_moves[best_index]
            else: # Other player
                best_move = board.random_move()


            board.make_move(best_move)
            # Prune MCTS tree so that root = board state
            mcts.prune_search_tree(board)
        
        if board.winner == player_ID:
            mc_player_ws += 1
    print("After running", games, "games, player 1 won", mc_player_ws)
# Assume MCTS optimal. See if using the D to pick moves is also optimal strategy.
def Test_the_D(player_ID):
    # Setup.
    k = 6 # board dimensions.
    s = 2 # time of each simulation, in seconds.
    games = 100 # Games played between the two.
    # Instances used.
    actor = Actor.Random_actor(k)
    board = hb.hex_board(k) # Actual game board
    mcts = MC.MCTS(0.1, board, actor)
    agent = Agent(0.2, [], k, 0.2, 2, 512, 200, 1) # Just used to test the D
    # Winner counter
    mc_player_ws = 0
    # Run the games
    starting_player=-1
    for i in range(games):
        print("========================\nGame number:", i,", Player",player_ID,"has won:",mc_player_ws)
        # Alternate starting turn
        if starting_player==-1:
            starting_player = 1
        else:
            starting_player = -1
        print("Starting player is player:", starting_player,"\n========================")
        # Reset mcts, all nodes removed.
        board.initialize_states(starting_player=starting_player)
        mcts.initialize(board)
        # Run the game
        while not board.game_over:
            # Run simulations, for both turns.
            mcts.simulate_timed(s, progress=0, verbose=True) # run hundreds of sims
            # player player_ID
            if board.player_turn==player_ID:
                all_moves = board.possible_moves_pos_2()           
                # Find best move - by visit counts
                D = agent.get_the_D(hash(board), all_moves, mcts.N_v) # normalized visit counts.
                best_move = all_moves[np.argmax(D)] # The highest index of D, is the best move.

            else: # Other player
                best_move = board.random_move()

            # Make the move
            board.make_move(best_move)
            # Prune MCTS tree so that root = board state
            mcts.prune_search_tree(board)
        
        if board.winner == player_ID:
            mc_player_ws += 1
    print("After running", games, "games, player",player_ID,"won", mc_player_ws)
# Full RL agent test, random actor.
def RL_agent_test_1():
    # Train in the end, random algorithm.
    learning_rate = 0.01
    k = 6
    layers = [60,60,60,60]             # 
    eps = 1                         # How random the rollouts are. Decay over the training.
    sim_time = 15                   # How accurate the training data is.
    RBUF_size = 5000                  # How much training data we can have.
    mbs = RBUF_size                        # How much data we can train on at once.
    actual_games = 200               # More => more training data.
    epochs = 20000                  # How much fitting we want, beware of the overfit.
    interval = 10                   # Saving away agents for TOPP
    
    # Create the agent, mode 1
    agent = Agent(learning_rate, layers, k, eps, sim_time, RBUF_size, mbs, mode=1)
    print("Agent created with a random actor.")
    
    # Train the agents actor
    agent.random_training_sim(actual_games, interval, epochs, verbose=True) # 100
    print("Agent is trained!") 
    
    # Test the agent vs a random agent!
    random = Actor.Random_actor(k)  # p1, a new random actor
    trained = agent.actor           # p2
    board = hb.hex_board(k)         # The board to play on
    p1_w = 0                        # Keep track of p2 victories
    games = 1000
    print("============================================")
    print("============== TIME TO FIGHT! ==============")
    print("============================================")
    print("Performance as p1:")
    for i in range(games): # Play games games
        # Reset the board. New game.
        board.initialize_states() 
        
        while board.game_over == False:
            # The player whos turn it is takes makes a move.
            if board.player_turn==-1: # The random player starts.
                move_dist = random.move_distribution(board.flatten_state())
            else: # The trained player
                if board.player_turn!=1:
                    raise Exception("No ones turn!")
                move_dist = trained.move_distribution(board.flatten_state())
            # Normalize the distribution.
            legal_moves =  np.multiply(move_dist, board.possible_moves) # Remove impossible moves.
            norm_moves = legal_moves / np.sum(legal_moves)

            # Completely greedy move.
            index = np.argmax(norm_moves) # Index of the best move, given a 1D board.
            row = index//k
            col = index%k

            # Make the move
            board.make_move((row, col))

        #print("Round",i+1," was won by player:", board.winner)
        if board.winner == 1:
            p1_w += 1
    print("After",games,"games. The trained player won:",p1_w)
    p2_w = 0 
    print("Performance as p2:")
    for i in range(games): # Play games games
        # Reset the board. New game.
        board.initialize_states() 
        
        while board.game_over == False:
            # The player whos turn it is takes makes a move.
            if board.player_turn==1: # The random player starts.
                move_dist = random.move_distribution(board.flatten_state())
            else: # The trained player
                if board.player_turn!=-1:
                    raise Exception("No ones turn!")
                move_dist = trained.move_distribution(board.flatten_state())
            # Normalize the distribution.
            legal_moves =  np.multiply(move_dist, board.possible_moves) # Remove impossible moves.
            norm_moves = legal_moves / np.sum(legal_moves)

            # Completely greedy move.
            index = np.argmax(norm_moves) # Index of the best move, given a 1D board.
            row = index//k
            col = index%k

            # Make the move
            board.make_move((row, col))

        #print("Round",i+1," was won by player:", board.winner)
        if board.winner == -1:
            p2_w += 1
    print("After",games,"games. The trained player won:",p2_w)
# Full RL agent test, NN actor.
def RL_agent_test_2():
    # Train as intended.
    learning_rate = 0.05                # Learning rate of NN
    k = 6                               # Size of board
    layers = [20,20,20,20,20]       # Structure of NN
    eps = 1                             # How random the rollouts are. Decay over the training.
    sim_time = 10                       # How accurate the training data is.
    RBUF_size = 512                     # How much training data we can have.
    mbs = 200                           # How much data we can train on at once, after each actual game.
    actual_games = 400                  # More => more training data.
    epochs = 20                         # How much fitting we want, beware of the overfit.
    topp_models = 5                     # No. Topp models stored away during training.
    interval = actual_games/topp_models # Saving away agents for TOPP
    activation = "tanh"  


    # Create the agent, mode 2
    agent = Agent(learning_rate, layers, k, eps, sim_time, RBUF_size, mbs, 2, activation=activation)
    print("Agent created with a NN actor.")
    
    # Train the agents actor
    agent.NN_training_sim(actual_games, interval, epochs, verbose=True, save_model=False) 
    print("Agent is trained!") 
    
    # Test the agent vs a random agent!
    random = Actor.Random_actor(k)  # p1, a new random actor
    trained = agent.actor           # p2
    board = hb.hex_board(k)         # The board to play on
    p1_w = 0                        # Keep track of p2 victories
    games = 1000
    print("============================================")
    print("============== TIME TO FIGHT! ==============")
    print("============================================")
    print("Performance as p1:")
    for i in range(games): # Play games games
        # Reset the board. New game.
        board.initialize_states() 
        
        while board.game_over == False:
            # The player whos turn it is takes makes a move.
            if board.player_turn==-1: # The random player starts.
                move_dist = random.move_distribution(board.flatten_state())
            else: # The trained player
                if board.player_turn!=1:
                    raise Exception("No ones turn!")
                move_dist = trained.move_distribution(board.flatten_state())
            # Normalize the distribution.
            legal_moves =  np.multiply(move_dist, board.possible_moves) # Remove impossible moves.
            norm_moves = legal_moves / np.sum(legal_moves)

            # Completely greedy move.
            index = np.argmax(norm_moves) # Index of the best move, given a 1D board.
            row = index//k
            col = index%k

            # Make the move
            board.make_move((row, col))

        #print("Round",i+1," was won by player:", board.winner)
        if board.winner == 1:
            p1_w += 1
    print("After",games,"games. The trained player won:",p1_w)
    p2_w = 0 
    print("Performance as p2:")
    for i in range(games): # Play games games
        # Reset the board. New game.
        board.initialize_states() 
        
        while board.game_over == False:
            # The player whos turn it is takes makes a move.
            if board.player_turn==1: # The random player starts.
                move_dist = random.move_distribution(board.flatten_state())
            else: # The trained player
                if board.player_turn!=-1:
                    raise Exception("No ones turn!")
                move_dist = trained.move_distribution(board.flatten_state())
            # Normalize the distribution.
            legal_moves =  np.multiply(move_dist, board.possible_moves) # Remove impossible moves.
            norm_moves = legal_moves / np.sum(legal_moves)

            # Completely greedy move.
            index = np.argmax(norm_moves) # Index of the best move, given a 1D board.
            row = index//k
            col = index%k

            # Make the move
            board.make_move((row, col))

        #print("Round",i+1," was won by player:", board.winner)
        if board.winner == -1:
            p2_w += 1
    print("After",games,"games. The trained player won:",p2_w)
# Generate and test agents for TOPP demo.
def TOPP_generate_agents_for_demo():
    '''Generate the 5x5 agents for topp demo.'''
    # Train as intended.
    learning_rate = 0.05                # Learning rate of NN
    k = 5                               # Size of board
    layers = [50, 50, 50]               # Structure of NN
    eps = 1                             # How random the rollouts are. Decay over the training.
    sim_time = 5                        # How accurate the training data is.
    RBUF_size = 512                     # How much training data we can have.
    mbs = 50                            # How much data we can train on at once, after each actual game.
    actual_games = 200                  # More => more training data.
    epochs = 100                        # How much fitting we want, beware of the overfit.
    topp_models = 5                     # No. Topp models stored away during training.
    interval = actual_games/topp_models # Saving away agents for TOPP
    activation = "tanh"  
    G = 100

    # Create the agent, mode 2
    agent = Agent(learning_rate, layers, k, eps, sim_time, RBUF_size, mbs, 2, activation=activation)
    
    # Train the agents actor
    agent_paths = agent.NN_training_sim(actual_games, interval, epochs, verbose=True, save_model=True) # 100
    print("Agent is trained!") 
    
    # Test the agent vs a random agent!
    random = Actor.Random_actor(k)  # p1, a new random actor
    trained = agent.actor           # p2
    board = hb.hex_board(k)         # The board to play on
    p1_w = 0                        # Keep track of p2 victories
    games = 1000
    print("==============================================")
    print("============== FIGHT VS RANDOM! ==============")
    print("==============================================")
    print("Performance as p1:")
    for i in range(games): # Play games games
        # Reset the board. New game.
        board.initialize_states() 
        
        while board.game_over == False:
            # The player whos turn it is takes makes a move.
            if board.player_turn==-1: # The random player starts.
                move_dist = random.move_distribution(board.flatten_state())
            else: # The trained player
                if board.player_turn!=1:
                    raise Exception("No ones turn!")
                move_dist = trained.move_distribution(board.flatten_state())
            # Normalize the distribution.
            legal_moves =  np.multiply(move_dist, board.possible_moves) # Remove impossible moves.
            norm_moves = legal_moves / np.sum(legal_moves)

            # Completely greedy move.
            index = np.argmax(norm_moves) # Index of the best move, given a 1D board.
            row = index//k
            col = index%k

            # Make the move
            board.make_move((row, col))

        #print("Round",i+1," was won by player:", board.winner)
        if board.winner == 1:
            p1_w += 1
    print("After",games,"games. The trained player won:",p1_w)
    p2_w = 0 
    print("Performance as p2:")
    for i in range(games): # Play games games
        # Reset the board. New game.
        board.initialize_states() 
        
        while board.game_over == False:
            # The player whos turn it is takes makes a move.
            if board.player_turn==1: # The random player starts.
                move_dist = random.move_distribution(board.flatten_state())
            else: # The trained player
                if board.player_turn!=-1:
                    raise Exception("No ones turn!")
                move_dist = trained.move_distribution(board.flatten_state())
            # Normalize the distribution.
            legal_moves =  np.multiply(move_dist, board.possible_moves) # Remove impossible moves.
            norm_moves = legal_moves / np.sum(legal_moves)

            # Completely greedy move.
            index = np.argmax(norm_moves) # Index of the best move, given a 1D board.
            row = index//k
            col = index%k

            # Make the move
            board.make_move((row, col))

        #print("Round",i+1," was won by player:", board.winner)
        if board.winner == -1:
            p2_w += 1
    print("After",games,"games. The trained player won:",p2_w)
    print("============================================")
    print("============== FIGHT IN TOPP! ==============")
    print("============================================")
    topp = TOPP(k)
    topp.stochastic_round_robin(agent_paths, G, interval=interval)
# Use saved data from random_training to test parameters
def train_on_saved_data():
    '''Used saved data from random_training to test learning rates.'''
    pass
# Run a single series between random actors to check for biases.
def test_single_series():
    k = 6
    no_games = 10000
    topp = TOPP(k)
    start_time = time.time()
    print("Pitting two random actors against each other for",no_games,"games, on a",k,"x",k,"board.")
    print("They should win equally many games, as they swap who goes first.")
    # Make two random agents, they should get 50/50 scores.
    random1 = Actor.Random_actor(k)
    random2 = Actor.Random_actor(k)
    pw1 = topp.single_series(random1, random2, no_games)
    #pw1 = topp.stochastic_single_series(random1, random2, no_games)
    print("Winrates were:", pw1,"/",no_games-pw1)
    print("That took:", time.time()-start_time)
# 
###############################################
############### Component tests ###############
###############################################
def random_walk_animate(animate=True):
    ''' Animate random walk to test board logic and displat'''
    # PARAMETERs
    board_size = 5 #k x k
    episode = []
    frame_delay = 500
    figsize = (15,15)
     
    # Board and display
    board = hb.hex_board(board_size)
    display = hd.hex_display(frame_delay, figsize)

    # Add first state, and create an episode
    episode.append(board.state)
    while not board.game_over:
        # Choose random possible move
        move = board.random_move()

        # Execute move.
        board.make_move(move)
        
        # Add to episode list, used to animate.
        episode.append(board.state)

    if animate:
        # Animate the episode!
        print(board.state)
        #print(board.edge_connections)
        display.animate_episode(episode, board_size)
    return board.winner
#
def random_walk_check_bias(num):
    p1 = 0
    p2 = 0
    for i in range(num):
        winner = random_walk_animate(animate=False)
        if winner == 1:
            p1 += 1
        else:
            p2 += 1
    print("Running,",num," random games, where P1 started each time:")
    print("P1:", p1, "\t=>", p1*100/(p1+p2), "% winrate")
    print("P2:", p2, "\t=>", p2*100/(p1+p2), "% winrate")
#
def hash_and_compare():
    dic={}
    board1 = hb.hex_board(10)
    board2 = hb.hex_board(10)
    print("#1: Same")
    dic[hash(board1)] = 1
    dic[hash(board2)] = 2
    print(dic)
    print("1:",hash(board1))
    print("2:",hash(board2))
    print("#2: 1 deviate")
    board1.make_move((1,1))
    dic[hash(board1)] = 3
    dic[hash(board2)] = 4
    print(dic)
    print("1:",hash(board1))
    print("2:",hash(board2))
    print("#3: 2 deviate")
    board2.make_move((2,2))
    dic[hash(board1)] = 5
    dic[hash(board2)] = 6
    print(dic)
    print("1:",hash(board1))
    print("2:",hash(board2))
    print("#4: same")
    board1.make_move((3,3))
    board2.make_move((3,3))
    board1.make_move((2,2))
    board2.make_move((1,1))
    dic[hash(board1)] = 7
    dic[hash(board2)] = 8
    print(dic)
    print("1:",hash(board1))
    print("2:",hash(board2))

    print("\n-----------------------\nPROBLEM")
    board3 = hb.hex_board(3)
    board4 = hb.hex_board(3)

    board3.make_move((2,2))
    board4.make_move((0,0))
    board4.make_move((0,1))
    board4.make_move((1,0))
    print(board3)
    print(board4)
    print("Is board3 == board4?", board3==board4)
  #
#
def test_max():
    l = [1, 2, 3, 4, 9, 5, 6, 7]
    i = 1
    def f(num):
        return num * i
    
    print("# 1 ")
    for i in range(1,3):
        print("max:",max(l, key=f))

    def g(s):
        if s =="ja":
            return 10
        else:
            return 0

    print("# 2 ")
    l_2 = ["hei", "på", "deg", "ja"]
    print("max 2:", max(l_2, key=g))
#
def test_copy_method():
    board1 = hb.hex_board(5)
    board2 = board1 # Is this a copy or the same one?
    board3 = deepcopy(board1)
    
    # 1 
    if board1 == board2:
        print("They were just made, should be equal.")
    else:
        raise Exception("They should be equal")

    # 2 make same moves in different patterns
    board1.make_move((3,3))
    board1.make_move((2,2))
    board1.make_move((4,4))
    board1.make_move((1,1))
    
    # 3 See what is what
    if board1 == board2:
        print("They are still equal, meaning they points to same object.")
    else:
        print("= operator works, making a copy with no need for a deep copy.")

    if board3==board1 or board3==board2:
        raise Exception("Board 3 should be a deepcopy...")
    else:
        print("deepcopy() makes a new, independent instance.")
#
def test_flatten():
    board = hb.hex_board(4)
    print(board.flatten_state())
    board.make_move((0,0))
    print(board.flatten_state())
    board.make_move((1,1))
    print(board.flatten_state())
    board.make_move((2,2))
    print(board.flatten_state())
    board.make_move((3,3))
    print(board.flatten_state())
#
def hex_board_child_test():
    k= 11
    board = hb.hex_board(k)
    possible_moves = board.possible_moves_pos()
    l = board.all_children_boards(possible_moves)
    print("For a "+str(k)+"x"+str(k)+" board, there should be " +str(k*k) +" possible moves, and therefor " +str(k*k) +" children:")
    print("Length of possible moves:", len(possible_moves))
    print("No. children generated:", len(l))
#
def test_possible_moves():
    board = hb.hex_board(6)
    moves = []
    while not board.game_over:
        print(board.possible_moves_pos())
        move = board.random_move()
        print("Chose this:", move)
        moves.append(move)
        board.make_move(move)
    print("Winner:", board.winner)
    print(board)

# Fresh
#new_MCTS_single_sims()
#new_MCTS_one_actual_game()
#RL_agent_test_1() # Random actor, save in end.
#RL_agent_test_2() # NN actor, train as we go.
#TOPP_generate_agents_for_demo()
#test_single_series()
#MCTS_optimality_test(player_ID=1)
#MCTS_optimality_test(player_ID=-1)
#Test_the_D(player_ID=1) # had about 90% wr as p2.
### Integrated tests
#random_walk_animate() 
#random_walk_check_bias(1000)
### Component TESTS
#MCTS_single_test()
#hex_board_child_test()
#test_max()
#test_copy_method()
#hash_and_compare()
### Single function tests
#test_flatten()
#test_possible_moves()
