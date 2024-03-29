import math
from tensorflow import keras
import numpy as np

from BasicClientActorAbs import BasicClientActorAbs
from actor import TOPP_Actor

class BasicClientActor(BasicClientActorAbs):

    def __init__(self, IP_address=None, verbose=True):
        self.series_id = -1
        BasicClientActorAbs.__init__(self, IP_address, verbose=verbose)         ##### TRIAL SCORE #####
        
        # RL 1 algorithm
        #self.actor = TOPP_Actor(keras.models.load_model("./power_model"))      # => 86/100 
        #self.actor = TOPP_Actor(keras.models.load_model("./power_model_2"))    # => 82/100 score in trial run.
        #self.actor = TOPP_Actor(keras.models.load_model("./power_model_3"))    # => 96/100 score in trial run. 87.0 in real run. [(2246887, 1, 28, 22), (2020, 2, 22, 28)]
        #self.actor = TOPP_Actor(keras.models.load_model("./power_model_4"))     # => 98/100 in trial run. Score was 80.5 in real run. [(2246887, 1, 15, 35), (2020, 2, 35, 15)]

        # RL2 algorighm
        #self.actor = TOPP_Actor(keras.models.load_model("./iron_man"))         # => 90/100 score in trial run.
        #self.actor = TOPP_Actor(keras.models.load_model("./iron_man_mk_2"))    # => 92/100 score in trial run.
        self.actor = TOPP_Actor(keras.models.load_model("./iron_man_mk_3"))    # => 96/100 score in trial run. 72 in actual run.


    # The shit
    def handle_get_action(self, state):
        """
        Here you will use the neural net that you trained using MCTS to select a move for your actor on the current board.
        Remember to use the correct player_number for YOUR actor! The default action is to select a random empty cell
        on the board. This should be modified.
        :param state: The current board in the form (1 or 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), where
        1 or 2 indicates the number of the current player.  If you are player 2 in the current series, for example,
        then you will see a 2 here throughout the entire series, whereas player 1 will see a 1.
        :return: Your actor's selected action as a tuple (row, column)
        """
        # 1 Make sure state is a np array.
        state = np.array(state)
        # 2 Get the possible moves from state:
        possible_moves = np.copy(state[1:]) # the first is just an ID
        k = np.sqrt(len(possible_moves))
        #print("==============================")
        #print("POSSBLE MOVES:", possible_moves)
        #print("k:", k)
        for i in range(len(possible_moves)):
            if possible_moves[i] == 0:
                possible_moves[i] = 1
            else:
                possible_moves[i] = 0
        # 3 Translate to my board representation.
        state[state==2] = -1
        # 4 Use model to get the distribution
        move_dist = self.actor.move_distribution(state)
        # 5 Remove illegal moves and normalize the distribution.
        legal_moves =  np.multiply(move_dist, possible_moves) 
        norm_moves = legal_moves / np.sum(legal_moves)
        # 6 Completely greedy move.
        index = np.argmax(norm_moves) # Index of the best move, given a 1D board.
        #print("INDEX:", index)
        row = int(index//k)
        col = int(index%k)
        next_move = (row, col)
        #print("NEXT MOVE:", next_move)
        #print("==============================")
        # 7 Return the move
        return next_move

    def handle_series_start(self, unique_id, series_id, player_map, num_games, game_params):
        """
        Set the player_number of our actor, so that we can tell our MCTS which actor we are.
        :param unique_id - integer identifier for the player within the whole tournament database
        :param series_id - (1 or 2) indicating which player this will be for the ENTIRE series
        :param player_map - a list of tuples: (unique-id series-id) for all players in a series
        :param num_games - number of games to be played in the series
        :param game_params - important game parameters.  For Hex = list with one item = board size (e.g. 5)
        :return

        """
        self.series_id = series_id
        #############################
        #
        #
        # YOUR CODE (if you have anything else) HERE
        #
        #
        ##############################

    def handle_game_start(self, start_player):
        """
        :param start_player: The starting player number (1 or 2) for this particular game.
        :return
        """
        self.starting_player = start_player
        #############################
        #
        #
        # YOUR CODE (if you have anything else) HERE
        #
        #
        ##############################

    def handle_game_over(self, winner, end_state):
        """
        Here you can decide how to handle what happens when a game finishes. The default action is to print the winner and
        the end state.
        :param winner: Winner ID (1 or 2)
        :param end_state: Final state of the board.
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        ##############################
        print("Game over, these are the stats:")
        print('Winner: ' + str(winner))
        print('End state: ' + str(end_state))

    def handle_series_over(self, stats):
        """
        Here you can handle the series end in any way you want; the initial handling just prints the stats.
        :param stats: The actor statistics for a series = list of tuples [(unique_id, series_id, wins, losses)...]
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("Series ended, these are the stats:")
        print(str(stats))

    def handle_tournament_over(self, score):
        """
        Here you can decide to do something when a tournament ends. The default action is to print the received score.
        :param score: The actor score for the tournament
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("Tournament over. Your score was: " + str(score))

    def handle_illegal_action(self, state, illegal_action):
        """
        Here you can handle what happens if you get an illegal action message. The default is to print the state and the
        illegal action.
        :param state: The state
        :param action: The illegal action
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("An illegal action was attempted:")
        print('State: ' + str(state))
        print('Action: ' + str(illegal_action))


if __name__ == '__main__':
    bsa = BasicClientActor(verbose=True)
    bsa.connect_to_server()
