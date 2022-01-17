import numpy as np
from typing import List

# Going to try to make a neural network that can handle multi-layer architecture
# Later
# Why doesn't it have a bias term?
class NeuralNetwork(object):
    def __init__(self, 
        num_input_nodes:int, 
        num_hidden_nodes:int, 
        num_output_nodes:int, 
        learning_rate:float
        )->None:
        """Initialize the neural networks.

        Args:
            num_input_nodes (int): Number of features
            num_hidden_nodes (int): the number of nodes in a hidden layer. Always have on 1 hidden layer.
            num_output_nodes (int): number of output nodes
            learning_rate (float): The step size for use in gradient descent
        """
        # Set number of nodes in input, hidden and output layers.
        self.num_input_nodes = num_input_nodes
        self.num_hidden_nodes = num_hidden_nodes
        self.num_output_nodes = num_output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.num_input_nodes**-0.5, 
                                       (self.num_input_nodes, self.num_hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.num_hidden_nodes**-0.5, 
                                       (self.num_hidden_nodes, self.num_output_nodes))
        self.learning_rate = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x : self.sigmoid(x)  # Replace 0 with your sigmoid calculation.
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid
                    
    def sigmoid(self, x)->float:
        # TODO: Return the result of calculating the sigmoid activation function
        #       shown in the lectures
        """Activation function.  That translates points to a probability space.
        
            There are other activation functions.  But this one uses the sigmoid
            equation (shown below), to convert linear combinations to values
            between 0 and 1.
        """
        return 1.0/(1+np.exp(-x))
    
    # def sigmoid_output_2_derivative(self,output)->float:
    #     # TODO: Return the derivative of the sigmoid activation function, 
    #     #       where "output" is the original output from the sigmoid function 
    #     return (self.sigmoid(output) * (1 - self.sigmoid(output)))
    #Not applying activation function here
    def sigmoid_output_2_derivative(self, output:float):
        """Derivative of the sigmoid activation function.

        Args:
            output ([float]): output value of the node being evaluated.

        Returns:
            float: derivate of the sigmoid function at the output value
        """
        return output * (1 - output)

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        # Send in features and targets for each 
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

        print(f"final weights:\n Input to Hidden: {self.weights_input_to_hidden} \n Hidden to Output {self.weights_hidden_to_output}")

    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = X.dot(self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = hidden_outputs.dot(self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # activation function is f(x) = x
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error - Replace this value with your calculations.
        error = final_outputs - y # Output layer error is the difference between desired target and actual output.

        # TODO: Calculate the hidden layer's contribution to the error
        #hidden_error = error * self.sigmoid_output_2_derivative(hidden_outputs) # original
        output_error_term = error * (self.weights_hidden_to_output.T) # activation function = f(x) = x, so derviative = 1
        
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        #output_error_term = hidden_error.dot(self.weights_hidden_to_output)
        hidden_error_term = output_error_term * self.sigmoid_output_2_derivative(hidden_outputs)
        
        # TODO: Add Weight step (input to hidden) and Weight step (hidden to output).
        # Weight step (input to hidden)
        delta_input = -np.dot(hidden_error_term.T, np.array([X]))  # DO I NEED THE NEGATIVE HERE?
        delta_weights_i_h += delta_input.T # Correct

        # Weight step (hidden to output) - NOT WORKING
        delta_output = -np.dot(np.array([error]),np.array([hidden_outputs]))
        delta_weights_h_o +=  delta_output.T
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += (1.0/n_records) * delta_weights_h_o * self.learning_rate # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += (1.0/n_records) * delta_weights_i_h * self.learning_rate # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = features.dot(self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = hidden_outputs.dot(self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer.  Activation function at this node is f(x) = x 
        print(f"Run final output: {final_outputs}")
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 100
learning_rate = 0.1
hidden_nodes = 2
output_nodes = 1
