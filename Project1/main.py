'''
Main file:
- for running the simulations.
- and setting parameters.

'''
#############################################################
######################### Imports ###########################
#############################################################
# Modules
from SW_peg_solitaire import SW, BT # The simworld
from RL_agent import Agent # The RL agent
import SW_tests as SWTest # Testing the visualization, and the animation
from SW_visualization import Display # Visualization of the board
# Other modules, not in use
import critic
import actor
import splitGD
import nn_critic
# Libraries
import numpy as np # Maths
import matplotlib.pyplot as plt # Graphing
import networkx as nx # Network graph
from matplotlib import animation # Animating network graphs
import time

def main():
    #############################################################
    ####################### PARAMETERS ##########################
    #############################################################
    epsilon_mode = 2            # Linear decay from e_0 to zero.
    initial_epsilon = 0.5       # Initial value of epsilon.
    critic_mode = 1             # 1 for table, 2 for NN.
    discount = 0.9              # Gamma.
    alpha_a = 0.4               # Learning rate of actor.
    alpha_c = 0.4              # Learning rate of critic.
    epoch = 1000                 # Number of episodes in epoch.
    lambda_a = 0.9              # Decay rate of actors eligibility. 
    lambda_c = 0.9              # Decay rate of critics eligibility. 
    board_type = BT.TRIANGLE    # Board type for the simworld.
    board_size = 5              # Board dimension.
    initial_holes = 1           # Initial no. holes. 1 is always centered.
    reward_mode = 0             # Reward regime. 0 is the best performer for now.
    layer_dimensions = [8]      # Number of hidden layers in nn, and size of each layer
    #############################################################
    ####################### PARAMETERS ##########################
    #############################################################
    ''' Play around with:
    #Nodes in dense layer.
    Activation functions, especially output.
    Learning rates: He used lr: 0.0001 for NN and 0.1 for table
    What seperates diamond and triangle? Coding error somewhere?
    Number of layers.

    Input size - calculate once.
    The game is over - use one method.
    '''
    '''Epsilon modes explained'''
    # mode 0 = zero
    # Mode 1 = constant 
    # Mode 2 = Linear decay over number of episodes
    # Mode 3 = Flat -> Linear
    # Mode 4 = Linear -> Flat, zero ending
    # Passing parameters as a list
    parameters = [
        epsilon_mode,
        initial_epsilon,
        critic_mode,
        discount,
        alpha_a,
        alpha_c,
        epoch,
        lambda_a,
        lambda_c,
        board_type,
        board_size,
        initial_holes,
        reward_mode,
        layer_dimensions]   
    
    # Some solutions to showcase:
    
    # Table
    ### 4_1 Diamond
    #parameters = [2, 0.5, 1, 0.8, 0.4, 0.4, 200, 0.9, 0.9, BT.DIAMOND, 4, 1, 0, [0]]
    ### 5_1 Triangle
    #parameters = [2, 0.5, 1, 0.8, 0.4, 0.4, 1000, 0.9, 0.9, BT.TRIANGLE, 5, 1, 0, [0]] #~ 79% (BEST)

    # NN 
    ### 4_1, Diamond
    #parameters = [2, 0.5, 2, 0.8, 0.4, 0.1, 200, 0.9, 0.9, BT.DIAMOND, 4, 1, 0, [16]]
    ### 5_1, Triangle
    #parameters = [2, 0.5, 2, 0.8, 0.4, 0.01, 600, 0.9, 0.9, BT.TRIANGLE, 5, 1, 0, [8]] 

    parameters = [2, 0.9, 2, 0.8, 0.4, 0.5, 1000, 0.9, 0.9, BT.TRIANGLE, 5, 1, 0, [3,3,3]] 

    ### TYPES of simulation
    #simulation_w_animation(parameters)
    simulation_wo_animation(parameters, 10)

    '''
    # Nightly simulations
    ### Combined best performers - needs testing
    #parameters = [2, 0.5, 1, 0.8, 0.6, 0.6, 1000, 0.9, 0.9, BT.TRIANGLE, 5, 1, 0]
    scores = [] 
    for i in range(1):
        parameters = [2, 0.5, 1, 0.8, 0.4, 0.4, 1000, 0.9, 0.9, BT.TRIANGLE, 5, 1, 0]
        scores.append(simulation_wo_animation(parameters , 500))
    print("-----------------------------------------")
    for i in range(len(scores)):
        print("Complete results")
        print("Mode:",i,":",scores[i],"%")
    '''


# For presentation of one set of parameters.
def simulation_w_animation(parameters):
    
    # Parameters
    epsilon_mode = parameters[0]
    e_0 = parameters[1]
    critic_mode = parameters[2]
    discount = parameters[3]
    alpha_a = parameters[4]
    alpha_c = parameters[5]
    epochs = parameters[6]
    lambda_a = parameters[7]
    lambda_c = parameters[8]
    board_type = parameters[9]
    board_size = parameters[10]
    initial_holes = parameters[11]
    reward_mode = parameters[12]
    layer_dimensions = parameters[13]

    # Create the agent
    rl_agent = Agent(critic_mode, discount, alpha_a, alpha_c,
     epochs, lambda_a, lambda_c, board_type, board_size, initial_holes, reward_mode, layer_dimensions)
    # Run the learning simulation
    
    
    if critic_mode == 1: # Table
        pegs_left = rl_agent.learning(e_0, epsilon_mode)
    else: # NN
        pegs_left = rl_agent.nn_learning(e_0, epsilon_mode)
    
    
    
    # Other plotting variables
    n_episodes = np.linspace(1,epochs,epochs)
    epsilons = np.zeros(epochs)
    for i in range(len(epsilons)):
        epsilons[i] = rl_agent.epsilon(epsilon_mode, e_0, i, epochs)

    # Linear regression
    coef = np.polyfit(n_episodes,pegs_left,1)
    poly1d_fn = np.poly1d(coef) 
    # poly1d_fn is now a function which takes in x and returns an estimate for y

    # Plot the results
    plt.plot(n_episodes, pegs_left, label="Pegs left")
    plt.plot(n_episodes, poly1d_fn(n_episodes), '--k', label="Average")
    plt.plot(n_episodes, epsilons, label = "epsilon")
    plt.title("Pegs left standing over the epochs of training")
    plt.legend(loc='upper left')
    plt.xlabel("Episode number [1]")
    plt.ylabel("Pegs left standing")
    plt.show()


    # Showcase final run: 🥳
    finale_episode = rl_agent.run()
    # Create display!
    display = Display()
    
    # Animation variables
    states = finale_episode
    # Build plot
    fig, ax = plt.subplots(figsize=(7,7))

    def update(num):
        '''
        Made to help animator animate the board.
        '''
        ax.clear()

        fig.gca().invert_yaxis()
        #fig.gca().invert_xaxis()

        state = states[num]
        # Produce a neighbourhood matrix
        neighbour_matrix = display.transform(state, board_type, board_size)
        # Use networkX to make a graph based on the neighbourhood matrix
        display.graph = nx.from_numpy_array(neighbour_matrix)
        # Color nodes according to state matrix
        color_map = display.color_nodes(state, board_type, board_size)
        # Try something new: fixed node-positions
        nodepos = display.node_pos(state, board_type, board_size)
        # Draw the current frame
        nx.draw(display.graph, node_color=color_map, with_labels=True,pos=nodepos)

    # Make animation
    ani = animation.FuncAnimation(fig, update, frames=(len(states)), interval=1000, repeat=True)
    plt.show()

# For testing a set of parameters over many simulations.
def simulation_wo_animation(parameters, num_sims):
    # Parameters
    epsilon_mode = parameters[0]
    e_0 = parameters[1]
    critic_mode = parameters[2]
    discount = parameters[3]
    alpha_a = parameters[4]
    alpha_c = parameters[5]
    epochs = parameters[6]
    lambda_a = parameters[7]
    lambda_c = parameters[8]
    board_type = parameters[9]
    board_size = parameters[10]
    initial_holes = parameters[11]
    reward_mode = parameters[12]
    layer_dimensions = parameters[13]

    successes = 0
    for i in range(num_sims):
        print("Progress: " + str(round(100*i/num_sims,2)) + "%")
        # Create the agent
        # Create the agent
        rl_agent = Agent(critic_mode, discount, alpha_a, alpha_c,
            epochs, lambda_a, lambda_c, board_type, board_size, initial_holes, reward_mode, layer_dimensions)
        # Run the learning simulation
        if critic_mode == 1: # Table
            pegs_left = rl_agent.learning(e_0, epsilon_mode)
        else: # NN
            pegs_left = rl_agent.nn_learning(e_0, epsilon_mode)
        # Test the policy
        finale_episode = rl_agent.run()
        if rl_agent.sim.pegs_left(finale_episode[-1]) == 1:
            print("Success!")
            successes+=1
    
    print("Hit-ratio: " + str(round(100*successes/num_sims,2)) + "%")
    return round(100*successes/num_sims,2)


#################
####   Run   ####
#################
main()
