import numpy as np
from typing import List

# Going to try to make a neural network that can handle multi-layer architecture
# Why doesn't it have a bias term?
class NeuralNetwork(object):
    def __init__(self, 
        num_input_nodes:int, 
        num_hidden_nodes:List[int], 
        num_output_nodes:int, 
        learning_rate:float
        )->None:
        """Initialize the neural networks.

        Args:
            num_input_nodes (int): Number of features
            num_hidden_nodes (List[int]): the number of nodes in a hidden layer. The length of
                the list, tells you how many hidden layers there are.  Each value at an index
                is the number of nodes in that hidden layer.
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
                    
    def sigmoid(self,x)->float:
        # TODO: Return the result of calculating the sigmoid activation function
        #       shown in the lectures
        return 1/(1+np.exp(-x))
    
    # def sigmoid_output_2_derivative(self,output)->float:
    #     # TODO: Return the derivative of the sigmoid activation function, 
    #     #       where "output" is the original output from the sigmoid function 
    #     return (self.sigmoid(output) * (1 - self.sigmoid(output)))
    #Not applying activation function here
    def sigmoid_output_2_derivative(self,output):
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


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = None # signals into hidden layer
        hidden_outputs = None # signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = None # signals into final output layer
        final_outputs = None # signals from final output layer
        
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
        error = None # Output layer error is the difference between desired target and actual output.
        
        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = None
        
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        output_error_term = None
        
        hidden_error_term = None
        
        # TODO: Add Weight step (input to hidden) and Weight step (hidden to output).
        # Weight step (input to hidden)
        delta_weights_i_h += None
        # Weight step (hidden to output)
        delta_weights_h_o += None
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += None # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += None # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = None # signals into hidden layer
        hidden_outputs = None # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = None # signals into final output layer
        final_outputs = None # signals from final output layer 
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 100
learning_rate = 0.1
hidden_nodes = 2
output_nodes = 1
