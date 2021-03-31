# Imports
import numpy as np
import tensorflow as tf
import tensorflow.keras as ker
from tensorflow.keras import layers

class Actor:
    '''The NN based actor class'''
    def __init__(self, alpha, layer_dimensions, input_size):
        # Critic data.
        # Weigts are stored as a list of tensors, one for each layer. 
        self.eligibility = []
        # TD error.
        self.delta = 0

        # RL parameters
        self.alpha = alpha      # Learning rate alpha.

        # NN parameters
        self.layer_dimensions = layer_dimensions    # List of hidden layer dimensions.
        self.n_layers = len(self.layer_dimensions)  # Number of hidden layers used.
        self.input_size = input_size                # Size of input (and output) required by NN.
        # Create the model
        self.model = self.initialize()
        
    # Initialize the NN (once for each epoch of episodes)
    def initialize(self):
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
        # Size is input-1: as we don't pass on player ID.
        model.add(ker.layers.Dense(self.input_size-1, activation='softmax', name="predictions")) # Softmax to get a probability distribution vector.

        # Compiling the model.
        model.compile(optimizer=ker.optimizers.SGD(learning_rate=(self.alpha)), loss=ker.losses.KLDivergence(), metrics=[tf.keras.metrics.KLDivergence()])
        model.summary()
        
        # Save the model structure as a png.
        ker.utils.plot_model(model, "Current_model.png", show_shapes=True)
        
        # Now the model is ready for fitting.
        return model

    def move_distribution(self, flat_state):
        '''Return the move distribution, numpy array'''
        return self.model.predict(flat_state)[0]

    def train(self, inputs, targets, mbs):
        '''Fit the model based on a random minibatch taken from the replay buffer. Inputs is a flattened board state with player ID,
            while targets is the D - generated distribution.'''
        print(inputs)
        print(targets)
        self.model.fit(inputs, targets, epochs=1, batch_size=mbs)

class Random_actor:
    '''Acts randomly'''
    def __init__(self, k):
        self.k = k
    
    def move_distribution(self, flat_state):
        '''Returns a random move distribution to mimmic the real mcts actor'''
        return np.random.uniform(0,1,(self.k**2,))