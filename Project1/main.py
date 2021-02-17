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
    epsilon_mode = 2            # 0 = zero. 1 = const. 2 = Linear decay. 3 = Flat -> Linear decay. 4 = decay rate.
    initial_epsilon = 0.5       # Initial value of epsilon.
    epsilon_decay = 0.95        # Decay rate of epsilon (only used for mode 4).
    critic_mode = 1             # 1 for table, 2 for NN.
    discount = 0.8              # Gamma / Discount rate.
    alpha_a = 0.4               # Learning rate of actor.
    alpha_c = 0.4               # Learning rate of critic.
    epoch = 1000                # Number of episodes in epoch.
    lambda_a = 0.9              # Decay rate of actors eligibility. 
    lambda_c = 0.9              # Decay rate of critics eligibility. 
    board_type = BT.TRIANGLE    # Board type for the simworld.
    board_size = 5              # Board dimension.
    initial_holes = [(0,0)]     # Additional holes. (row, col). Empty => middle hole.
    reward_mode = 0             # Reward regime. 0 is the best performer for now.
    layer_dimensions = [4,4]    # Number of hidden layers in nn, and size of each layer
    frame_delay = 1000          # How long a frame is displayed before moving on. [ms]
    display_mode = 1            # 1 = display final policy, 0 = no dispalys.
    # Passing parameters as a list
    parameters = [
        epsilon_mode,
        initial_epsilon,
        epsilon_decay,
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

    #############################################################
    ######################## SHOWCASE ###########################
    #############################################################
    
        # TABLE
        # 4_1 Diamond
    #parameters = [2, 0.5, 0.95, 1, 0.8, 0.4, 0.4, 200, 0.9, 0.9, BT.DIAMOND, 4, [], 0, []]             # (8 sec, 100%)
    
        # 5_1 Triangle
    #parameters = [2, 0.5, 0.95, 1, 0.8, 0.4, 0.4, 600, 0.9, 0.9, BT.TRIANGLE, 5, [], 0, []]            # (24 sec, 98%)

        # NEURAL NET 
        # 4_1 Diamond
    #parameters = [2, 0.5, 0.95, 2, 0.8, 0.4, 0.1, 40, 0.9, 0.9, BT.DIAMOND, 4, [], 0, [4, 4]]          # (40 sek, 96%)
    #parameters = [2, 0.5, 0.95, 2, 0.8, 0.4, 0.1, 200, 0.9, 0.9, BT.DIAMOND, 4, [], 0, [2,2]]          # (3.5 min, 100%)
    
        # 5_1 Triangle
    #parameters = [2, 0.5, 0.95, 2, 0.8, 0.4, 0.01, 400, 0.9, 0.9, BT.TRIANGLE, 5, [], 0, [3, 3, 3]]    # (5 min, 84%)
    #parameters = [2, 0.5, 0.95, 2, 0.8, 0.4, 0.01, 600, 0.9, 0.9, BT.TRIANGLE, 5, [], 0, [8]]          # (8 min, 90%)

    #############################################################
    ####################### SIMULATION ##########################
    #############################################################
    ### TYPES of simulation
    if display_mode == 1:
        simulation_w_animation(parameters, frame_delay)
    else: # Simulates a number of epochs, calculating success rate.
        simulation_wo_animation(parameters, 20) 


# For presentation of one set of parameters with graphs and animations.
def simulation_w_animation(parameters, frame_delay):
    '''
    Simulates one epoch given parameters. Then displays the learned solution,
     using frame_delay as 1/(frames per millisecond).
    '''
    # Parameters
    epsilon_mode = parameters[0]
    e_0 = parameters[1]
    epsilon_decay = parameters[2]
    critic_mode = parameters[3]
    discount = parameters[4]
    alpha_a = parameters[5]
    alpha_c = parameters[6]
    epochs = parameters[7]
    lambda_a = parameters[8]
    lambda_c = parameters[9]
    board_type = parameters[10]
    board_size = parameters[11]
    initial_holes = parameters[12]
    reward_mode = parameters[13]
    layer_dimensions = parameters[14]

    # Create the agent
    rl_agent = Agent(critic_mode, discount, alpha_a, alpha_c,
     epochs, lambda_a, lambda_c, board_type, board_size, initial_holes, reward_mode, layer_dimensions)
    
    
    # Run the learning simulation.
    if critic_mode == 1: # Table
        pegs_left = rl_agent.learning(e_0, epsilon_mode, epsilon_decay)
    else: # NN
        pegs_left = rl_agent.nn_learning(e_0, epsilon_mode, epsilon_decay)
    
    
    # Plotting the learning over time.
    n_episodes = np.linspace(1,epochs,epochs)
    epsilons = np.zeros(epochs)
    for i in range(len(epsilons)):
        epsilons[i] = rl_agent.epsilon(epsilon_mode, e_0, epsilon_decay, i, epochs)

    # Linear regression - average score.
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
        # Clear the board.
        ax.clear()
        # Invert y-axis, as draw uses opposite y than what I intended.
        fig.gca().invert_yaxis()
        # Extract single state.
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
    ani = animation.FuncAnimation(fig, update, frames=(len(states)), interval=frame_delay, repeat=True)
    plt.show()

# For testing a set of parameters over many simulations.
def simulation_wo_animation(parameters, num_sims):
    # Parameters
    epsilon_mode = parameters[0]
    e_0 = parameters[1]
    epsilon_decay = parameters[2]
    critic_mode = parameters[3]
    discount = parameters[4]
    alpha_a = parameters[5]
    alpha_c = parameters[6]
    epochs = parameters[7]
    lambda_a = parameters[8]
    lambda_c = parameters[9]
    board_type = parameters[10]
    board_size = parameters[11]
    initial_holes = parameters[12]
    reward_mode = parameters[13]
    layer_dimensions = parameters[14]

    successes = 0
    for i in range(num_sims):
        print("Progress: " + str(round(100*i/num_sims,2)) + "%")
        # Create the agent
        # Create the agent
        rl_agent = Agent(critic_mode, discount, alpha_a, alpha_c,
            epochs, lambda_a, lambda_c, board_type, board_size, initial_holes, reward_mode, layer_dimensions)
        # Run the learning simulation
        start = time.time()
        
        if critic_mode == 1: # Table
            pegs_left = rl_agent.learning(e_0, epsilon_mode, epsilon_decay)
        else: # NN
            pegs_left = rl_agent.nn_learning(e_0, epsilon_mode, epsilon_decay)
        # Test the policy
        end = time.time()
        print("Training time:", end - start)

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
