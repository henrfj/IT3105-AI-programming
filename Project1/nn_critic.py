'''
'''
# Libraries
import tensorflow as tf
from random import uniform
import tensorflow.keras as ker

# Modules
import splitGD as SGD

class NN_Critic:
    def __init__(self, discount, alpha_c, lambda_c, n_layers, input_size):
        # Using V(s): Key is state, returns value of state. Actor passes state to critic(*), 
        # who returns calculated TD error for given state.
        
        # Critic data.
        # Weigts are stored as a list, from 0->N. 
        self.eligibility = {}
        self.delta = 0
        #self.expected_value = {} # Why do I keep this list?

        # RL parameters
        self.discount = discount
        self.alpha_c = alpha_c
        self.lambda_c = lambda_c

        # NN parameters
        self.n_layers = n_layers # Number of hidden layers used.
        self.input_size = input_size

        # Create the initial splittable network.
        self.split_model = self.initialize_V()

    def initialize_V(self):
        # Sequential model
        model = ker.Sequential()
        # Input layer
        model.add(ker.Input(shape=(self.input_size, )))
        # A number of hidden layers
        for i in range(self.n_layers):
            model.add(ker.layers.Dense(i))
        # Compiling the model
        model.compile(optimizer=ker.optimizers.SGD(learning_rate=(self.alpha_c)), loss=ker.losses.MeanSquaredError(), metrics=['accuracy'])
        model.summary()
        # Not the model is ready for fitting, but we need to split it.
        sgd_model = SGD.SplitGD(model, self)
        return sgd_model

    def initialize_e(self, state, mode):
        ''' 
        Reset at start of each simulation.
        '''
        self.eligibility = {}
    
    def update_V(self, input_state, target):
        '''
        INPUT: the current state, transformed to input-format
        TARGET: the previouslycalculated target of the input.

        **Bootstrapping**: we go back over all state pairs, and 
        based on the newly found reward -> delta, we update weights
        who were active this episode, based on eligibility.
        '''
        self.split_model.fit(input_state, target)

        '''
        We are re-training every feature-target for each step we take. But for each step we get a new reward and a new delta.
        So based on the delta of the newest states, we train the funcapp on all previous states as well.
        (state, V*) are kept while we go through the episode. 
        For every new move we make, we re-train the (state, V*) based on the new delta - but still the old target.
        TODO: Is this correct?
        '''

    def update_delta(self, V_star, V_theta):
        ''' Calculates new delta, using NN prediction'''
        
        self.delta = V_star - V_theta
        return self.delta
    
    def update_gradients(self, gradients):
        '''
        Function called inside of splitGD in order to update the gradients, using eligibility.
        About updating e here:
        
        For RL with NN, eligibilities are closely connected to each weight, rather than each state.
        Eligibilities is the glue between NN and state-table, as you tell the NN what weights were
        useful in predicting the last  couple of states, so that they can get some reward/punishment for the current prediction.
        
        '''
        # Add gradients to eligibilities of weights. Or, initialize the eligibilities.
        # TODO: Do we need to start at 1?
        for i in range(len(gradients)):
            try:
                self.eligibility[i] = self.eligibility[i] + gradients[i]
            except:
                self.eligibility[i] = gradients[i]

        ## Update gradients
        for i in range(len(gradients)):
            gradients[i] += self.alpha_c * self.delta * self.eligibility[i]

        # Decay
        # TODO: Decay now, or at some other time?
        for i in range(len(self.eligibility)):
                self.eligibility[i] *= self.discount * self.lambda_c 
        
        # New and improved.
        return gradients
