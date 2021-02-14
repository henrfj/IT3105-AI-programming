'''
Unit tests for RL methods
'''

import numpy as np

board_size = 5
'''
Generate a size*size list with nodes.
Nodes are initially filled with pegs
'''
# Create a empty (zero-filled) size x size matrix.
state = np.zeros((board_size, board_size), int)
# Fill the board with default size, being filled with pegs.
for row in range(0,board_size):
    for col in range(0, board_size):
        state[row][col] = 1

'''Now initialize the holes'''
# Number of holes inserted. (Initial_holes >= 1)
n = 1
# Initial hole created close to the center
state[((board_size)//2)][((board_size-1)//2)] = 0



# Keeping track of layers of triangle
i = 0
inputs = np.zeros(board_size*board_size)
for row in range(board_size):
    for col in range(board_size):
        inputs[i] = state[row][col]
        i+=1

print(state)
print(inputs)
