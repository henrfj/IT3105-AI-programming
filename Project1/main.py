'''
Main file:
- running the simulations
- setting parameters

'''

from SW_peg_solitaire import SW, BT
from RL_agent import Agent
import SW_tests as SWTest


def main(parameters):
        
    # Running
    #TODO: run the simulations after setting parameters

        
    testing()

   


def testing():
    # 1
    #SWTest.create_diamond_board()
    # 2
    #SWTest.create_triangle_board()
    # 3
    #SWTest.find_child_states()
    # 5
    #SWTest.convert_to_matrix()
    # 6
    #SWTest.pins_left_test()
    # 7
    #SWTest.visualization_test()
    # 8
    #SWTest.animation_test()
    # 9
    SWTest.animation_test_actions()

parameters = []
main(parameters)