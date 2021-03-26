# My own modules
import hex_board as hb
import hex_display as hd
from copy import deepcopy

# Others modules
import matplotlib.pyplot as plt # Plotting
import networkx as nx # Network graph
from matplotlib import animation # Animating network graphs
import numpy as np # Efficient arrays


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
    board1 = hb.hex_board(10)
    board2 = hb.hex_board(10)
    # 1
    if board1 == board2:
        print("They are equals.")

    # 2
    dic = {}
    dic[board1] = 100
    print(dic)
    dic[board2] = 50
    print(dic)

    # 3
    board3 = hb.hex_board(13)
    board4 = hb.hex_board(9)
    l = [board3, board4]
    if board1 in l:
        print("This is wrong...")
    l.append(board1)
    if board2 in l:
        print("Board 1 = board 2!")

    # 4
    board1.make_move((3,3))
    board1.make_move((2,2))
    board1.make_move((4,4))
    board1.make_move((1,1))
    
    board2.make_move((4,4))
    board2.make_move((2,2))
    board2.make_move((3,3))
    board2.make_move((1,1))

    if board1 == board2:
        print("They are still equals after doing some moves, in different order.")

    # 5 Has board 1 changed now? => yes.
    dic[board1] = 77
    print(dic)
    dic[board2] = 22
    print(dic)

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

    

### TESTS
test_max()
#test_copy_method()
#hash_and_compare()
#random_walk_animate() 
#random_walk_check_bias(1000)