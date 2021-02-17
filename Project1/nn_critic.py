'''
The Class of the NN based critic.
'''
# Libraries
import tensorflow as tf
from random import uniform
import tensorflow.keras as ker
from tensorflow.keras import layers
import numpy as np

# Modules
import splitGD as SGD

class NN_Critic:
    '''The class of the NN based Critic'''
    def __init__(self, discount, alpha_c, lambda_c, layer_dimensions, input_size):
        # Critic data.
        # Weigts are stored as a list of tensors, one for each layer. 
        self.eligibility = []
        # TD error.
        self.delta = 0

        # RL parameters
        self.discount = discount    # Gamma, discount rate.
        self.alpha_c = alpha_c      # Learning rate alpha.
        self.lambda_c = lambda_c    # Eligibility decay lambda.

        # NN parameters
        self.layer_dimensions = layer_dimensions    # List of hidden layer dimensions.
        self.n_layers = len(self.layer_dimensions)  # Number of hidden layers used.
        self.input_size = input_size                # Size of input required by NN.
        
    # Initialize the NN (once for each epoch)
    def initialize_V(self):
        '''Initialize NN based on n-layers and input size.'''
       
        # Sequential model.
        model = ker.Sequential()
       
        # Input layer.
        model.add(ker.Input(shape=(self.input_size, )))
        
        # A number of hidden layers
        for i in range(self.n_layers):
            # Adds dense layers of different dimensions.
            model.add(ker.layers.Dense(self.layer_dimensions[i], activation='tanh'))
        
        # Output layer.
        # Activation "tanh" works best. 
        model.add(ker.layers.Dense(1, activation='tanh'))

        # Compiling the model.
        model.compile(optimizer=ker.optimizers.SGD(learning_rate=(self.alpha_c)), loss=ker.losses.MeanSquaredError(), metrics=['accuracy'])
        model.summary()
        
        # Save the model structure as a png.
        ker.utils.plot_model(model, "Current_model.png", show_shapes=True)
        
        # Now the model is ready for fitting, but we need to split it.
        self.split_model = SGD.SplitGD(model, self)
        
    # Initialize eligibility (once for each episode) 
    def initialize_e(self):
        ''' 
        Reset at start of each simulation.
        Is a list of tensors, one layer for each layer of the NN.
        '''
        # Will be initialized once we have the shape of gradients.
        self.eligibility = []

    # Update the NN by fitting a single feature and target.
    def update_V(self, input_state, target):
        '''
        INPUT: the current state, transformed to input-format
        TARGET: the previously calculated target of the input.

        **Bootstrapping**: we go back over all state pairs, and 
        based on the newly found reward -> delta, we update weights
        who were active this episode, based on eligibility.
        '''
        # Verbosity = level of printing
        # vfrac = fraction of data kept as training data.
        self.split_model.fit(input_state, target, vfrac=0, verbosity=0)

    # Update delta based on TD-error algorithm.
    def update_delta(self, V_star, V_theta):
        ''' Calculates new delta, using NN prediction'''
        # V_star and V_theata are both outputs of the NN, 
        # and are of shape (1,1) - we would like a scalar delta.
        self.delta = (V_star - V_theta)[0][0]
        return self.delta
    
    # Update gradients, used by splitGD.modify_gradients()
    def update_gradients(self, gradients):
        '''
        Function called inside of splitGD in order to update the gradients, using eligibility.
        About updating e here:
        
        For RL with NN, eligibilities are closely connected to each weight, rather than each state.
        Eligibilities is the glue between NN and state-table, as you tell the NN what weights were
        useful in predicting the last  couple of states, so that they can get some reward/punishment for the current prediction.
        
        Gradients is a list of tensors, one for each layer. Contains the gradients of each weight related to each layer.
        '''
        # 1
        # Make eligibilities into same shape as gradients, filled with 0.
        if len(self.eligibility) == 0:
            self.eligibility = []
            for i in range(len(gradients)):
                self.eligibility.append(tf.fill(tf.shape(gradients[i]), 0.0))

        # 2
        ## Update gradients.
        
        # 2.1
        ## First add gradients to eligibilities of weights.
        for i in range(len(gradients)):
            self.eligibility[i] = tf.math.add(self.eligibility[i], gradients[i])
        
        # 2.2
        ## Update gradients based on eligibility.
        for i in range(len(gradients)):
            #self.eligibility[i] * self.alpha_c * self.delta
            ts = tf.math.multiply(self.eligibility[i], self.alpha_c * self.delta)
            # ts = tf.reshape(ts, tf.shape(self.eligibility[i]))
            # gradients[i] += ts
            gradients[i] = tf.math.add(gradients[i], ts)

        # 2.3 
        ## Decay the eligibilities.
        for i in range(len(self.eligibility)):
            self.eligibility[i] = tf.math.multiply(self.eligibility[i], self.discount * self.lambda_c)

        # 3
        # Return updated gradients.
        return gradients


