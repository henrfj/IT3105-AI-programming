'''
Main file:
- running the simulations
- setting parameters

'''

from SW_peg_solitaire import *
from rl_engine import *
import SW_tests as SWTest



class Main:
    def __init__(self, parameters):
        self.parameters = parameters

        # Running
        #TODO: run the simulations after setting parameters

   


def testing():
    # 1
    SWTest.create_diamond_board()
    # 2
    SWTest.create_triangle_board()
    
testing()