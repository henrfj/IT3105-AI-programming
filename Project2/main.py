# My own modules
import hex_board as hb
import hex_display as hd
import mcts as MC
import actor as Actor
from RL_agent import Agent

# Others modules
import matplotlib.pyplot as plt # Plotting
import networkx as nx # Network graph
from matplotlib import animation # Animating network graphs
import numpy as np # Efficient arrays
from copy import deepcopy # Deep copies


def random_walk_animate(animate=True):
    ''' Animate random walk to test board logic and displat'''
    # PARAMETERs
    board_size = 11 #k x k
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

def test_flatten():
    board = hb.hex_board(4)
    print(board.flatten_state())
    board.make_move((0,0))
    board.make_move((1,1))
    board.make_move((2,2))
    board.make_move((3,3))
    print(board.flatten_state())

def MCTS_single_test():
    # Setup.
    k = 6 # board dimensions
    s = 2 # time in seconds
    # Instances used.
    actor = Actor.Random_actor(k)
    board = hb.hex_board(k)
    mcts = MC.MCTS(0.1, board, actor)
    
    # Run simulations, look for result
    mcts.simulate_timed(s, progress=0) 
    print("No. Parents", len(list(mcts.children.keys())))
    print("N-keys:", len(mcts.N.keys()))
    print("Q-keys:", len(list(mcts.Q.keys())))
    print("N_v-keys:", len(mcts.N_v.keys()))
    print("Max edges:", len(list(mcts.children.keys())) * (k*k)) # 
    print("Max nodes:", len(list(mcts.children.keys())) * k*k + 1) # +1 is the root

def hex_board_child_test():
    k= 11
    board = hb.hex_board(k)
    possible_moves = board.possible_moves_pos()
    l = board.all_children_boards(possible_moves)
    print("For a "+str(k)+"x"+str(k)+" board, there should be " +str(k*k) +" possible moves, and therefor " +str(k*k) +" children:")
    print("Length of possible moves:", len(possible_moves))
    print("No. children generated:", len(l))

def MCTS_one_actual_game_test(animate=True):
    ''' Uses statistics from MCTS directly to make moves'''
    
    # Setup.
    k = 6 # board dimensions
    s = 2 # time of each simulation, in seconds
    frame_delay = 500
    figsize = (8,8)
    # Instances used.
    actor = Actor.Random_actor(k)
    board = hb.hex_board(k) # Actual game board
    mcts = MC.MCTS(0.1, deepcopy(board), actor)
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
        
        # Run a new sim
        mcts.simulate_timed(s, progress=0, verbose=True)
        possible_moves = board.possible_moves_pos()


    print("The winner of the actual game is player "+str(board.winner)+"!")
    print("The MCTS simulation tree grew to this size:")
    print("Parent nodes:", len(mcts.children.keys()))
    print("Total nodes:", len(mcts.N.keys()), ", out of", len(mcts.children.keys())*k**2,' "possible" nodes.')
    print("The winning board looked like this:\n",board.state)
    if animate:
        display.animate_episode(episode, k)

def RL_agent_test():
    # Params
    learning_rate = 0.1
    layers = [5,5,5]
    k = 3
    eps = 0.2
    sim_time = 1
    RBUF_size = 512
    mbs = 300
    
    # Create the agent
    agent = Agent(learning_rate, layers, k, eps, sim_time, RBUF_size, mbs)
    print("Agent created with a NN actor.")
    
    # Train the agents actor
    agent.run_training_sim(100, 10, verbose=True)
    print("Agent is trained!")
    
    # Test the agent vs a random agent!
    random = Actor.Random_actor(k)  # p1, a new random actor
    trained = agent.actor           # p2
    board = hb.hex_board(k)         # The board to play on
    p2_w = 0                        # Keep track of p2 victories
    games = 100
    print("============================================")
    print("============== TIME TO FIGHT! ==============")
    print("============================================")
    
    for i in range(games): # Play games games
        # Reset the board. New game.
        board.initialize_states() 
        
        while board.game_over == False:
            # The player whos turn it is takes makes a move.
            if board.player_turn==1: # The random player starts.
                move_dist = random.move_distribution(board.flatten_state())
            else: # The trained player
                if board.player_turn!=2:
                    raise Exception("No ones turn!")
                move_dist = trained.move_distribution(board.flatten_state())
            # Normalize the distribution.
            legal_moves =  np.multiply(move_dist, board.possible_moves) # Remove impossible moves.
            norm_moves = legal_moves / np.sum(legal_moves)

            # Completely greedy move.
            index = np.argmax(norm_moves) # Index of the best move, given a 1D board.
            row = index%k
            col = index%k

            # Make the move
            board.make_move((row, col))

        print("Round",i+1," was won by player:", board.winner)
        if board.winner == 2:
            p2_w += 1
    
    print("After",games,"games. The trained player won:",p2_w)


### Integrated tests
RL_agent_test()
#MCTS_one_actual_game()
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
