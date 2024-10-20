import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt

# Display The Image, taken from online
def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)

# LAYER LAYER LAYER LAYER LAYER LAYER LAYER LAYER LAYER LAYER LAYER LAYER
# LAYER LAYER LAYER LAYER LAYER LAYER LAYER LAYER LAYER LAYER LAYER LAYER
# LAYER LAYER LAYER LAYER LAYER LAYER LAYER LAYER LAYER LAYER LAYER LAYER
# LAYER LAYER LAYER LAYER LAYER LAYER LAYER LAYER LAYER LAYER LAYER LAYER
# LAYER LAYER LAYER LAYER LAYER LAYER LAYER LAYER LAYER LAYER LAYER LAYER

# Dense layer for general use
class Layer_Dense:

    # Layer initialization for inputs, neurons, and regularization
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons

        # Set regularization strength for self
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    # Forward pass, we include training just for modularity with other functions
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass, calculating the derivitives
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * \
                             self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * \
                            self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    # Get the neurons
    def getNeurons(self) :
        return self.n_neurons
    
    # Get the inputs
    def getInputs(self) :
        return self.n_inputs

    # Retrieve the layers parameters
    def get_parameters(self):
        return self.weights, self.biases

    # Set weights and biases in layer
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

# Dense layer for general use
class Recurrent_Layer_Dense:

    # Layer initialization for inputs, neurons, and regularization
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.hiddenWeights = np.zeros((1, n_neurons))
        self.hiddenWeights += 1
        self.biases = np.zeros((1, n_neurons))
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons

        # Set regularization strength for self
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    # Forward pass, we include training just for modularity with other functions
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, (self.weights * self.hiddenWeights)) + self.biases

        # Change the hidden weights based on inputs
        if not training:
            for outputChange in self.output:
                self.hiddenWeights *= outputChange

    # Backward pass, calculating the derivitives
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * \
                             self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * \
                            self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    # Get the neurons
    def getNeurons(self) :
        return self.n_neurons
    
    # Get the inputs
    def getInputs(self) :
        return self.n_inputs

    # Retrieve the layers parameters
    def get_parameters(self):
        return self.weights, self.biases

    # Set weights and biases in layer
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

# Dropout layer for helping model generalize
class Layer_Dropout:

    # Init
    def __init__(self, dropout):
        # Store rate, we invert it as for example for dropout
        # of 0.1 we need success rate of 0.9
        self.rate = 1 - dropout

    # Forward pass
    def forward(self, inputs, training):
        # Save input values
        self.inputs = inputs

        # If not in the training mode - return the original values, nothing else
        if not training :
            self.output = inputs.copy()
            return
        
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate,
                           size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    # Backward pass
    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask

# Used for input layer in Model
class Layer_Input:

    # Forward pass
    def forward(self, inputs, training):
        self.output = inputs

# ACTIVATION ACTIVATION ACTIVATION ACTIVATION ACTIVATION ACTIVATION ACTIVATION
# ACTIVATION ACTIVATION ACTIVATION ACTIVATION ACTIVATION ACTIVATION ACTIVATION
# ACTIVATION ACTIVATION ACTIVATION ACTIVATION ACTIVATION ACTIVATION ACTIVATION
# ACTIVATION ACTIVATION ACTIVATION ACTIVATION ACTIVATION ACTIVATION ACTIVATION
# ACTIVATION ACTIVATION ACTIVATION ACTIVATION ACTIVATION ACTIVATION ACTIVATION

# ReLU activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs

# Softmax activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)

        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):

        # Create uninitialized array with size of dvalues
        self.dinputs = np.empty_like(dvalues)

        # Enumerate over outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # Flatten array using reshape
            single_output = single_output.reshape(-1, 1)
            # Jacobian matrix
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient so we can add it to the gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)


# Sigmoid activation
class Activation_Sigmoid:

    # Forward pass
    def forward(self, inputs, training):
        # Save input and calculate/save output
        # of the sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    # Backward pass
    def backward(self, dvalues):
        # Derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output
        
    # Calculate predictions for outputs
    def predictions(self, outputs): 
        return (outputs > 0.5) * 1

class Activation_Linear:

    # Forward pass
    def forward(self, inputs, training):
        # Remember vars
        self.inputs = inputs
        self.output = inputs

    # Backward pass
    def backward(self, dvalues):
        # derivitive is 1, so we just do the dvalues again cuz 1 times anything is itself
        self.dinputs = dvalues.copy()

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs

# OPTIMIZER OPTIMIZER OPTIMIZER OPTIMIZER OPTIMIZER OPTIMIZER OPTIMIZER OPTIMIZER
# OPTIMIZER OPTIMIZER OPTIMIZER OPTIMIZER OPTIMIZER OPTIMIZER OPTIMIZER OPTIMIZER
# OPTIMIZER OPTIMIZER OPTIMIZER OPTIMIZER OPTIMIZER OPTIMIZER OPTIMIZER OPTIMIZER
# OPTIMIZER OPTIMIZER OPTIMIZER OPTIMIZER OPTIMIZER OPTIMIZER OPTIMIZER OPTIMIZER
# OPTIMIZER OPTIMIZER OPTIMIZER OPTIMIZER OPTIMIZER OPTIMIZER OPTIMIZER OPTIMIZER

# SGD optimizer
class Optimizer_SGD:

    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If we use momentum
        if self.momentum:

            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights
                # The array doesn't exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * \
                             layer.dweights
            bias_updates = -self.current_learning_rate * \
                           layer.dbiases

        # Update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

# Adagrad optimizer
class Optimizer_Adagrad:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

# RMSprop optimizer
class Optimizer_RMSprop:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

# Adam optimizer
class Optimizer_Adam:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum  with current gradients
        layer.weight_momentums = self.beta_1 * \
                                 layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \
                               layer.bias_momentums + \
                               (1 - self.beta_1) * layer.dbiases
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2

        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         weight_momentums_corrected / \
                         (np.sqrt(weight_cache_corrected) +
                             self.epsilon)
        layer.biases += -self.current_learning_rate * \
                         bias_momentums_corrected / \
                         (np.sqrt(bias_cache_corrected) +
                             self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

# LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS
# LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS
# LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS
# LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS
# LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS

# Common loss class
class Loss:

    # Set and remember the trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    # Regularization loss calculation
    def regularization_loss(self):

        # 0 by default
        regularization_loss = 0

        # Calculate for all trainable layers
        for layer in self.trainable_layers:

            # L1 regularization - weights
            # calculate only when factor greater than 0
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                                    np.sum(np.abs(layer.weights))

            # L2 regularization - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                                    np.sum(layer.weights * \
                                            layer.weights)

            # L1 regularization - biases
            # calculate only when factor greater than 0
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                                    np.sum(np.abs(layer.biases))

            # L2 regularization - biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                                    np.sum(layer.biases * \
                                            layer.biases)

        return regularization_loss

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y, *, include_regularization=False):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        # If just data loss - return it
        if not include_regularization:
            return data_loss

        # Return loss
        return data_loss, self.regularization_loss()

    # Calculate the accumulated loss across steps
    def calculate_accumulated(self, *, include_regularization=False):
        # Calculate loss mean
        data_loss = self.accumulated_sum / self.accumulated_count

        # If just data, return it
        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()

    # Resets the accumulation vars for a new epoch and for function above
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Binary cross-entropy loss
class Loss_BinaryCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        # Return losses
        return sample_losses

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues -
                         (1 - y_true) / (1 - clipped_dvalues)) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Mean Squared Error Loss
class Loss_MeanSquaredError(Loss): #L2 Loss

    # Forward pass
    def forward(self, y_pred, y_true):

        # Calculate loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)

        # Return 
        return sample_losses
    
    # Bakward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We will use the first sample to count them since they all same size I believe
        outputs = len(dvalues[0])

        # Gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalize
        self.dinputs = self.dinputs / samples

# Mean Absolute Error Loss
class Loss_MeanAbsoluteError(Loss): #L1 Loss

    # Forward pass
    def forward(self, y_pred, y_true):

        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

        # Return 
        return sample_losses
    
    # Bakward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We will use the first sample to count them since they all same size I believe
        outputs = len(dvalues[0])

        # Gradient on values
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize
        self.dinputs = self.dinputs / samples

# ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY
# ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY
# ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY
# ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY ACCURACY

# Common Accuracy Class
class Accuracy:

    # Calculates an acuracy given predictions, and the ground truth values
    def calculate(self, predictions, y_truth):

        # Get comparison results
        comparisons = self.compare(predictions, y_truth)

        # Calculate an accuracy
        accuracy = np.mean(comparisons)

        # Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        # Return accuracy
        return accuracy
    
    # Calculate accumulated accuracy for the steps
    def calculate_accumulated(self):
        # Calculate loss mean
        accuracy = self.accumulated_sum / self.accumulated_count

        return accuracy

    # Reset vars for accumulation like we did in loss
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
# Accuracy Calculation for regression Model
class Accuracy_Regression(Accuracy):

    def __init__(self):
        # Create an attribute called precision
        self.precision = None

    # Calculates precision based on ground truth values
    def init(self, y_truth, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y_truth) / 250

    # Compares predictions to the ground truth values
    def compare(self, predictions, y_truth):
        return np.absolute(predictions - y_truth) < self.precision

# Accuracy Class for a Classification Model 
class Accuracy_Categorical(Accuracy):
    
    def __init__(self, *, binary=False):
        # Binary Mode?!?
        self.binary = binary

    # Does nothing, but we still need it since it is called in our Model 
    def init(self, y_truth):
        pass

    # Compares predictions to the ground truth values
    def compare(self, predictions, y_truth):
        if not self.binary and len(y_truth.shape) == 2:
            y_truth = np.argmax(y_truth, axis=1)
        
        return predictions == y_truth

# MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL 
# MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL 
# MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL 
# MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL MODEL 
class Model:

    def __init__(self):
        # Create a list of network objects
        self.layers = []
        # Softmax classifiers output object
        self.softmax_classifier_output = None

    # Adding object to the neural network model
    def add(self, layer) :
        self.layers.append(layer)

    # Sets Loss and Optimizer
    # the * means that the loss and optimizer are keywords or something like that
    def set(self, *, loss=None, optimizer=None, accuracy=None) :
        if loss is not None:
            self.loss = loss    

        if optimizer is not None:
            self.optimizer = optimizer

        if accuracy is not None:
            self.accuracy = accuracy

    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        # Default if batch size is None
        train_steps = 1

        # Init accuracy
        self.accuracy.init(y)

        if batch_size is not None:
            train_steps = len(X) // batch_size
            # Dividing rounds down, if there is remaining we will add 1 to make sure we include everything
            if train_steps * batch_size < len(X):
                train_steps += 1

        # Main Training Loop
        for epoch in range(1, epochs+1):

            # Print epoch number
            print(f'epoch: {epoch}')

            # Reset Accumulated values for the steps
            self.loss.new_pass()
            self.accuracy.new_pass()
            
            for step in range(train_steps):

                # If batch size is not set - train using one step and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y

                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                    
                # Here we perform the forward pass
                output = self.forward(batch_X, training=True)

                # Calculate loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss

                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                # Backward Pass now
                self.backward(output, batch_y)

                # Optimize the layers after the backward pass has been completed
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                # Print Summary
                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                        f'acc: {accuracy:.3f}, ' +
                        f'loss: {loss:.3f} (' +
                        f'data_loss: {data_loss:.3f}, ' +
                        f'reg_loss: {regularization_loss:.3f}), ' +
                        f'lr: {self.optimizer.current_learning_rate}')
            
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} ()' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate}')

            # If there is the validation data
            if validation_data is not None:
                
                # Here we evalulate the model
                # The * is to seperate the data into multiple params
                self.evaluate(*validation_data, batch_size=batch_size)
        

    # Forward Pass for the Neural Network Model
    def forward(self, X, training) :
        # Call forward method on the input layer
        # This will set the output property that the first layer in "prev" object is expecting
        self.input_layer.forward(X, training)

        # Now we call forward for all the other layers inside the model
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # The var layer is now our last object in the list, so we want to return that layer's output
        return layer.output

    # backward Pass for the Neural Network Model
    def backward(self, output, y):
        
        # If it is softmax classifier
        if self.softmax_classifier_output is not None:
            # First call backward method on the combined activation/loss and set dinputs property
            self.softmax_classifier_output.backward(output, y)

            # Since we will call backward method of the last layer which is softmax
            # As we combined activation/loss, set dinputs in this object
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            # Call backward method going through all objects but the last in reversed order with dinputs as parameters
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            # Return so we don't go on
            return
            
        # First we call backward method on the loss so we can get its dinputs
        self.loss.backward(output, y)

        # Call backward method going through all layers in reverse order passing dinputs as the parameters
        for layer in reversed(self.layers) :
            layer.backward(layer.next.dinputs)


    # Finalizes the building of the model, 
    # And lets each layer be notified of its previous layers
    def finalize(self):

        # Create and set input layer
        self.input_layer = Layer_Input()

        # Count all the objects
        layer_count = len(self.layers)

        # Init the trainable layers to empty array
        self.trainable_layers = []

        # Iterate the objects
        for i in range(layer_count) :

            # If its the first layer, the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            # All layers except for first and last fall here
            elif i < (layer_count - 1):
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            # The last layer - the next object is the loss
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss # Loss function for finalizing
                self.output_layer_activation = self.layers[i] # Last Layer for Activation

            # If the layer contains the attuributes "weights" we know it is a tunable or trainable layer
            # If so we just add it to a list
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
        
        # Update loss object with trainable layers if we have a loss
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        # If output activation is Softmax and loss function is Categorical Cross Entropy, we want to combine them for a faster gradient calculations and better performance of Model
        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossentropy):
            # Create object of combined activation and loss functions
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()

    def evaluate(self, X_val, y_val, *, batch_size=None):
        # Validation steps set to 1 originally
        validation_steps = 1

        # Adjust the batch size
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size

            # Test if we need to add one more
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        # Reset Accumulated values for the steps
        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):

            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]

            # Forward pass
            output = self.forward(batch_X, training=False)

            # Calculate the loss
            self.loss.calculate(output, batch_y)

            # Get predictions, than calculate the accuracy
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        #Get data from the steps
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        # Print
        print(f'validation, ' +
            f'acc: {validation_accuracy:.3f}, ' +
            f'loss: {validation_loss:.3f}')

    def predict(self, X, * , batch_size=None):
        prediction_steps = 1

        # Same stuff as before to get the correct amount of steps to fit the entire batch
        if batch_size is not None:
            prediction_steps = len(X) // batch_size

            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        #Output of predictions
        output = []

        # Iterate over the steps
        for step in range(prediction_steps):

            # If batch size is not set, train using one step and the entire dataset
            if batch_size is None:
                batch_X = X

            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]

            # Perform forward pass
            batch_output = self.forward(batch_X, training=False)

            # Append it to the outputs list
            output.append(batch_output)

        # Return a stacked version of the outputs
        return np.vstack(output) # This will return a confidence array of all the outputs essentiall, or at least for classification model

    def query(self) :
        print(self.layers)

    def printModelImage(self) :
        fig = plt.figure(figsize=(12, 12))
        ax = fig.gca()
        ax.axis('off')
        i = 0
        modelLayers = []
        for l in self.layers :
            if(isinstance(l, Layer_Dense)) :
                modelLayers.append(l.getInputs())
                if(i == len(self.layers)-2) :
                    modelLayers.append(l.getNeurons())
            i = i + 1
        draw_neural_net(ax, .1, .9, .1, .9, modelLayers)
        fig.savefig('nn.png')
        plt.show()
        plt.clf()

    # Gets the parameters from all the layers in the model
    def get_parameters(self) :
        
        # Create empty array of parameters
        parameters = []
        
        # Iterate over the trainable layers and get parameters and add them to the list
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        
        # Return
        return parameters

    # Updates the model with new parameters
    def set_parameters(self, parameters):

        # We iterate over the parameters and the layers, updating each layer with the set of parameters
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    # Saves the parameters to a file with specific name
    def save_parameters(self, path):
        # Open the file in binary-write mode and save parameters to it
        with open(path, 'wb') as file:
            pickle.dump(self.get_parameters(), file)

    # Loads the parameters into the layers of the model
    def load_parameters(self, path):
            
        # Open the file in the binary-read mode, load weights and biases and update the trainable layers with them
        with open(path, 'rb') as file:
            self.set_parameters(pickle.load(file))

    # Saves the Entire Model
    def save(self, path):
        
        # Make a deep copy of current model instance
        model = copy.deepcopy(self)

        # Reset accumulated values just in case this is called after training
        model.loss.new_pass()
        model.accuracy.new_pass()

        # Remove data from input and gradients from the loss 
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        # For each layer we want to remove the inputs outputs and dinputs properties
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        # Open the file using python pickle in binary-write mode than write the stuff to the file to save the Model
        with open(path, 'wb') as file:
            pickle.dump(model, file)

    # Loads the Entire Model
    @staticmethod
    def load(path):

        # Open file in binary-read mode to load the model from the file
        with open(path, 'rb') as file:
            model = pickle.load(file)
        
        # Return the newly loaded model
        return model