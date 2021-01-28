'''
Node class:
- Basic building block of the board
'''
from enum import Enum

class Status(Enum):
    UNUSED = -1
    EMPTY = 0
    PEG = 1
    


class Node:
    '''Node class, for each node in the board.
    '''

    def __init__(self, status=Status.PEG):
        # Defalt status is pegged
        self.status = status

    def set_status(self, newStatus):
        if newStatus in [Status.EMPTY, Status.PEG, Status.UNUSED]:
            self.status = newStatus
        else:
            raise Exception("Trying to set illegal node-status")

    def get_status(self):
        return self.status