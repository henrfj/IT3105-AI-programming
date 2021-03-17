# My own modules
import hex_board as hb
import hex_display as hd

# Others modules
import matplotlib.pyplot as plt # Plotting
import networkx as nx # Network graph
from matplotlib import animation # Animating network graphs
import numpy as np # Efficient arrays


def random_walk_animate(animate=True):
    ''' Animate random walk to test board logic and displat'''
    # PARAMETERs
    board_size = 10 #k x k
    episode = []
    frame_delay = 1
    figsize = (15,15)
     
    # Board and display
    board = hb.hex_board(board_size)
    display = hd.hex_display(frame_delay, figsize)

    # Add first state, and create an episode
    episode.append(board.state)
    while not board.game_over:
        # Choose random possible move
        possible_moves = board.child_states() # A dictionary of {moves : pos}
        index = int(np.random.randint(0, len(possible_moves))) # Random index
        next_state = possible_moves[index][0] # The next state
        pos = possible_moves[index][1] # Position of moved piece

        # Execute move
        board.make_move(next_state, pos)
        # Add to episode list
        episode.append(board.state)

    if animate:
        # Animate the episode!
        print(board.state)
        print(board.edge_connections)
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


#random_walk_animate() q
random_walk_check_bias(50000)