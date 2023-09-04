import numpy as np
import scipy as sp

class Layer:
    """
    Layer of a neural network, performs basic steps like forward and backward propagation
    """
    def __init__(self,
                 input_size: int,
                 size: int,
                 weights: np.array,
                 activation_function: str,
                 alpha: float):
        """

        :param input_size:
        :param size:
        :param weights: The initial weights, can be set to None to initialize them automatically
        :param activation_function:
        :param alpha:
        """

        self.h = None  # The value of the layer after activation function
        self.z = None  # Value of layer before activation function
        self.input_value = None
        self.intercept = None
        self.input_size = input_size
        self.size = size

        #Sets initial weights like specified in the slides. This should make exploding values less likely
        if weights is not None:
            assert weights.shape == (self.size, self.input_size), \
                f"initial weights should be {self.size}x{self.input_size}"
            self.weights = weights
        else:
            self.initialize_weights()

        #Sets activation function and derivative based on input
        if activation_function == "tanh":
            self.activation_function = np.tanh
            self.activation_function_derivative = lambda x: np.diag((1 - np.tanh(x) ** 2).ravel())
        elif activation_function == "sigmoid":
            self.activation_function = sp.special.expit
            self.activation_function_derivative = lambda x: np.diag((sp.special.expit(x)*(1-sp.special.expit(x))).ravel())
        elif activation_function == "relu":
            self.activation_function = lambda x: np.maximum(x, 0)
            self.activation_function_derivative = lambda x: np.diag(np.heaviside(x, 1).ravel())
        else:
            self.activation_function = lambda x: x
            self.activation_function_derivative = lambda x: np.identity(len(x))

        self.alpha = alpha

        # initializes the derivative of weights and intercept with zero matrix
        self.del_weights = np.zeros(self.weights.shape)
        self.del_intercept = np.zeros(self.intercept.shape)

    def initialize_weights(self) -> np.array:
        """Initializes weights with normally distributed values"""
        self.weights = np.random.normal(0, 2/(self.size+self.input_size), (self.size, self.input_size))
        self.intercept = np.random.normal(0, 2/(self.size+self.input_size), (self.size,1))

    def forward_propagate(self, input_value: np.array):
        # Calculates the forward propagation by multiplying the weights with the input and adding the intercept.
        self.input_value = input_value
        self.z = np.matmul(self.weights, self.input_value) + \
                 np.matmul(self.intercept, np.ones((1, 1)))

        # Calculates the layer value with the activation function
        self.h = self.activation_function(self.z)

        return self.h

    def backward_propagate(self, derivative: np.array):
        """

        :param derivative: A numpy array of the derivative of the cross-entropy loss in regard to the value of the layer
        :return: A numpy array of the derivative of the cross-entropy loss in regard to the value of the previous layer
        """

        tmp = self.activation_function_derivative(self.z)  # Derivative of the activation function
        delz = np.matmul(derivative, tmp)  # Derivative of loss to z

        self.del_intercept += delz.transpose()

        delW = np.kron(np.identity(self.size), self.input_value.transpose())  # Derivative of z over weights W
        self.del_weights += np.reshape(np.matmul(delz, delW), self.weights.shape)
        return np.matmul(delz, self.weights)

    def update(self):
        """
        Updates the weights
        :return: nothing
        """
        self.intercept = self.intercept - self.alpha * self.del_intercept
        self.weights = self.weights - self.alpha * self.del_weights

        # Sets initial derivatives to zero
        self.del_intercept = np.zeros(self.intercept.shape)
        self.del_weights = np.zeros(self.weights.shape)
