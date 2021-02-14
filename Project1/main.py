'''
Main file:
- running the simulations
- setting parameters

'''
#############################################################
######################### Imports ###########################
#############################################################
# Modules
from SW_peg_solitaire import SW, BT # The simworld
from RL_agent import Agent # The RL agent
import SW_tests as SWTest # Testing the visualization, and the animation
from SW_visualization import Display # Visualization of the board
# Libraries
import numpy as np # Maths
import matplotlib.pyplot as plt # Graphing
import networkx as nx # Network graph
from matplotlib import animation # Animating network graphs

def main():
    #############################################################
    ####################### PARAMETERS ##########################
    #############################################################
    '''Epsilon modes explained'''
    # mode 0 = zero
    # Mode 1 = constant 
    # Mode 2 = Linear decay over number of episodes
    # Mode 3 = Flat -> Linear
    # Mode 4 = Linear -> Flat, zero ending
    epsilon_mode = 2        # 2
    initial_epsilon = 0.5   # 0.5
    critic_mode = 1         # 1
    discount = 0.80         # 0.80
    alpha_a = 0.40          # 0.40
    alpha_c = 0.40          # 0.40
    epoch = 1000            # Episodes per epoch
    lambda_a = 0.90         # 0.90
    lambda_c = 0.90         # 0.90
    board_type = BT.DIAMOND
    board_size = 4
    initial_holes = 1
    reward_mode = 0
    #############################################################
    ####################### PARAMETERS ##########################
    #############################################################
    
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
        reward_mode]   
    

    # Some solutions to showcase:
    ### Solves the 4_1 Diamond; table
    #parameters = [2, 0.5, 1, 0.8, 0.4, 0.4, 200, 0.9, 0.9, BT.DIAMOND, 4, 1, 0]
    ### Solved the 5_1 Triangle; table ~ 82%
    #parameters = [2, 0.5, 1, 0.8, 0.4, 0.4, 1000, 0.9, 0.9, BT.TRIANGLE, 5, 1, 2] ~ 82%
    #parameters = [2, 0.5, 1, 0.8, 0.4, 0.4, 1000, 0.9, 0.9, BT.TRIANGLE, 5, 1, 0] ~ 79% (BEST)
    ### Combined best performers - needs testing
    #parameters = [2, 0.5, 1, 0.8, 0.6, 0.6, 1000, 0.9, 0.9, BT.TRIANGLE, 5, 1, 0]


    # Test
    #parameters = [2, 0.5, 1, 0.8, 0.5, 0.5, 1000, 0.9, 0.9, BT.TRIANGLE, 5, 1, 0]
    ### TYPES of simulation
    #simulation_w_animation(parameters)
    #simulation_wo_animation(parameters, number)

    
    # Nightly simulations
    scores = [] 
    for i in range(1):
        parameters = [2, 0.5, 1, 0.8, 0.4, 0.4, 1000, 0.9, 0.9, BT.TRIANGLE, 5, 1, 0]
        scores.append(simulation_wo_animation(parameters , 500))
    print("-----------------------------------------")
    for i in range(len(scores)):
        print("Complete results")
        print("Mode:",i,":",scores[i],"%")
    

# For presentation of one set of parameters.
def simulation_w_animation(parameters):
    
    # Parameters
    epsilon_mode = parameters[0]
    e_0 = parameters[1]
    value_mode = parameters[2]
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

    # Create the agent
    rl_agent = Agent(value_mode, discount, alpha_a, alpha_c, epochs, lambda_a, lambda_c, board_type, board_size, initial_holes, reward_mode)
    # Run the learning simulation
    pegs_left = rl_agent.learning(e_0, epsilon_mode)
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
    value_mode = parameters[2]
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

    successes = 0
    for i in range(num_sims):
        print("Progress: " + str(round(100*i/num_sims,2)) + "%")
        # Create the agent
        rl_agent = Agent(value_mode, discount, alpha_a, alpha_c, epochs, lambda_a, lambda_c, board_type, board_size, initial_holes, reward_mode)
        # Run the learning simulation
        pegs_left = rl_agent.learning(e_0, epsilon_mode)
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
