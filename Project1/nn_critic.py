'''
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

        # V is initialized each epoch

    def initialize_V(self):
        
        # Sequential model.
        model = ker.Sequential()
       
        # Input layer.
        model.add(ker.Input(shape=(self.input_size, )))
        
        # A number of hidden layers
        for i in range(self.n_layers):
            model.add(ker.layers.Dense(self.input_size, activation='tanh'))
        
        # Output layer.
        # TODO: Linear activation function?
        model.add(ker.layers.Dense(1, activation='tanh'))

        # Compiling the model.
        model.compile(optimizer=ker.optimizers.SGD(learning_rate=(self.alpha_c)), loss=ker.losses.MeanSquaredError(), metrics=['accuracy'])
        model.summary()
        ker.utils.plot_model(model, "my_sequential_model.png", show_shapes=True)
        # Now the model is ready for fitting, but we need to split it.
        self.split_model = SGD.SplitGD(model, self)
        
    def initialize_e(self):
        ''' 
        Reset at start of each simulation.
        '''
        # Will be initialized once we have the shape of gradients.
        self.eligibility = []

    def update_V(self, input_state, target):
        '''
        INPUT: the current state, transformed to input-format
        TARGET: the previouslycalculated target of the input.

        **Bootstrapping**: we go back over all state pairs, and 
        based on the newly found reward -> delta, we update weights
        who were active this episode, based on eligibility.
        '''
        # Verbosity = level of printing
        self.split_model.fit(input_state, target, verbosity=0)

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
            ts = tf.math.multiply(self.eligibility[i], self.alpha_c * self.delta[0][0])
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


'''
# 1
# Turn gradients into one long list og dV/dw_i
grad = [] # All weight-specific gradients dV/dw
for layer in range(len(gradients)): # Run through all layers
    try:
        for node in range(len(gradients[layer])): # Run through all nodes in layer
            try:
                for weight in range(len(gradients[layer][node])): # Run throug all the nodes
                    grad.append(gradients[layer][node][weight].numpy())
            
            except: # gradients[layer][node] is a scalar. This is the input layer (16, 1)          
                grad.append(gradients[layer][node].numpy())
    
    except: # gradients[layer] is a scalar. This is the output weight (1, )
        grad.append(gradients[layer].numpy())
    
# 2
for i in range(len(grad)):
    try:
        self.eligibility[i] = self.eligibility[i] + grad[i]
    except:
        self.eligibility[i] = grad[i]

## Update gradients.
for i in range(len(grad)):
    grad[i] += self.alpha_c * self.delta * self.eligibility[i]

# Decay eligibilities.
for i in range(len(self.eligibility)):
        self.eligibility[i] *= self.discount * self.lambda_c
'''