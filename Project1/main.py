'''
Main file:
- running the simulations
- setting parameters

'''
# Modules
from SW_peg_solitaire import SW, BT
from RL_agent import Agent
import SW_tests as SWTest
# Libraries
import numpy as np
import matplotlib.pyplot as plt

def main(parameters):
        
    # Running
    #TODO: run the simulations after setting parameters    
    testing()


def testing():
    # 1
    #SWTest.animation_test_actions()
    # 2
    epsilon_mode = 2
    value_mode = 1
    discount = 0.4
    alpha_a = 0.4
    alpha_c = 0.4
    epochs = 5000
    lambda_a = 0.9
    lambda_c = 0.9
    board_type = BT.DIAMOND
    board_size = 4
    initial_holes = 7
    rl_agent = Agent(value_mode, discount, alpha_a, alpha_c, epochs, lambda_a, lambda_c, board_type, board_size, initial_holes)
    # Run the learning simulation
    pegs_left = rl_agent.learning(epsilon_mode)
    n_episodes = np.linspace(1,epochs,epochs)
    
    # Bug, negative PI values.
    print("PI:")
    print(min(rl_agent.actor.PI.values()))
    print("V:")
    print(min(rl_agent.critic.V.values()))

    # Linear regression
    coef = np.polyfit(n_episodes,pegs_left,1)
    poly1d_fn = np.poly1d(coef) 
    # poly1d_fn is now a function which takes in x and returns an estimate for y

    # Plot the results
    plt.plot(n_episodes, pegs_left, label="Pegs left")
    plt.plot(n_episodes, poly1d_fn(n_episodes), '--k', label="Average")
    plt.plot(n_episodes, rl_agent.epsilon(n_episodes, epochs, 2), label = "epsilon")
    plt.title("Pegs left standing over the epochs of training")
    plt.legend(loc='upper left')
    plt.xlabel("Episode number [1]")
    plt.ylabel("Pegs left standing")
    plt.show()

parameters = []
main(parameters)