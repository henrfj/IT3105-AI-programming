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
    SWTest.animation_test_actions()

    # 2

parameters = []
main(parameters)