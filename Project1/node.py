'''
Node class:
- Basic building block of the board
'''
from enum import Enum

class Status(Enum):
    EMPTY = 0
    PEG = 1


class Node:
    '''Node class, for each node in the board.
    '''

    def __init__(self):
        # Defalt status is pegged
        self.status = Status.PEG
        # Keeping track of neighbours
        self.neighbours = []


    def add_neighbour(self, neighbour):
        # Add a new neighbour, being another node
        # Neighbour are Nodes connected on the board.
        self.neighbours.append(neighbour)

        #TODO: add some kind of directional value to neighbours

    def get_neighbours(self):
        # Getter for neighbours of node
        return self.neighbours

    def set_status(self, newStatus):
        if newStatus in [Status.EMPTY, Status.PEG]:
            self.status = newStatus
        else:
            raise Exception("Trying to set illegal node-status")

    def get_status(self):
        return self.status