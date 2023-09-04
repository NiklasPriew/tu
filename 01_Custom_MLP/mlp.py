import numpy as np
from layer import Layer


class Mlp:
    """
    Manages the different layers of the neural network and deals with the correct data transfer.
    """
    def __init__(self,
                 x:np.array,
                 n_hidden_layers: int,
                 hidden_layer_depth: int | list,
                 activation_function: str,
                 train_type: str,
                 alpha: float,
                 classes: list):
        """

        :param x: The numpy array of the exogenuous variables. Only the shape of this is relevant
        :param n_hidden_layers: The number of hidden layers of the neural network
        :param hidden_layer_depth: The size of each hidden layer. This can be an integer or a list. If an integer is used each hidden layer has the same size
        :param activation_function: The activation function chosen. Can be 'tanh', 'sigmoid' or 'relu'. For every other string the identity function gets used
        :param train_type: The type of training. 0 is batch training, every other integer sets the size of the mini batch.
        :param alpha: The learning rate of the training
        :param classes: A list of every possible class.
        """
        self.n_hidden_layers = n_hidden_layers
        if type(hidden_layer_depth) == int:
            #transforms the input integer to a list
            self.hidden_layer_depth = [hidden_layer_depth] * self.n_hidden_layers
        else:
            self.hidden_layer_depth = hidden_layer_depth
        self.activation_function = activation_function
        self.alpha = alpha

        #Initialises the list of layers with the first hidden layer
        self.layers = [Layer(x.shape[1], size=self.hidden_layer_depth[0], weights=None,
                             activation_function=self.activation_function, alpha=self.alpha)]
        self.classes = classes

        #Appends the other hidden layers
        for i in range(1, self.n_hidden_layers):
            self.layers.append(
                Layer(input_size=self.hidden_layer_depth[i - 1], size=self.hidden_layer_depth[i], weights=None,
                      activation_function=self.activation_function, alpha=self.alpha))

        #Appends a layer for the final output
        self.layers.append(Layer(input_size=self.hidden_layer_depth[-1], size=len(classes), weights=None,
                                 activation_function=self.activation_function, alpha=self.alpha))

        self.n = train_type

    def forward_propagation(self, x):
        """
        :param x: a numpy array representing an observation in the exogenuous variables
        :return: returns the value of the last layer without applying softmax
        """
        tmp = x
        for layer in self.layers:
            tmp = layer.forward_propagate(tmp)
        return tmp

    def backward_propagation(self, deriv):
        """
        :param deriv: The derivative \partial L/\partial h
        """
        tmp = deriv
        for layer in self.layers[::-1]:
            tmp = layer.backward_propagate(tmp)

    def update(self):
        for layer in self.layers:
            layer.update()

    def softmax(self, v):
        """
        Calculate the softmax of input
        """
        # Invariant to constant shift. This makes sure the values do not explode as much
        e_x = np.exp(v - np.max(v))
        return e_x / np.sum(e_x)

    def fit(self, x_train, y_train, max_iterations):
        """

        :param x_train: The exogenuous values used for training
        :param y_train: The endogenuous values used for training, consisting of the same values provided to classes
        :param max_iterations: Sets the maximum number of iterations for the fitting
        :return: Nothing
        """
        if self.n == 0:
            self.n = len(y_train)
        prev_weights = None
        for j in range(max_iterations):
            choice = np.random.permutation(len(y_train))[:self.n]
            for index in choice:
                xi = x_train[index, :].reshape((len(x_train[index, :]), 1))
                h = self.forward_propagation(xi)
                result = self.softmax(h)
                #Calculate the initial derivative used for back propagation
                # https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
                derivative = result - np.array((self.classes == y_train[index])).reshape(len(self.classes), 1)
                self.backward_propagation(derivative.transpose())
            self.update()
            
            current_weights = np.concatenate([layer.weights.ravel() for layer in self.layers])

            #Stops training if weights do not change very much
            if prev_weights is not None and np.allclose(current_weights, prev_weights):
                break
            
            prev_weights = current_weights

    def predict(self, x_test):
        """
        Predicts class of observation from exogeniuous variables provided, can be provided with multiple observations
        at the same time
        :param x_test: Numpy Array
        :return: list with predicted classes
        """
        results = []
        for i in range(len(x_test)):
            xi = x_test[i, :].reshape((len(x_test[i, :]), 1))
            tmp = self.softmax(self.forward_propagation(xi))
            index = np.argmax(tmp)
            results.append(index)
        return results
