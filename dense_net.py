import numpy
import numpy as np
import tensorflow as tf

# class to construct and evaluate a fully connected neural network
class dense_net(object): 
    def __init__(self,layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        self.__name__ = 'swish'
        num_layers = len(layers) 
        xavier_stddev = np.sqrt(2/(layers[0] + layers[-1]))
        for i in range(0,num_layers-1):
            W = tf.Variable(tf.truncated_normal([layers[i],layers[i+1]],stddev= np.sqrt(2/(layers[i] + layers[i+1]))), dtype=tf.float32)
            b = tf.Variable(tf.zeros([1,layers[i+1]], dtype=tf.float32), dtype=tf.float32)
            self.weights.append(W)
            self.biases.append(b)
            
    
    def evaluate(self, X):
            H = X
            for i in range(0,len(self.layers)-2):
                W = self.weights[i]
                b = self.biases[i]
                H = tf.tanh(tf.matmul(H, W) + b)
            W = self.weights[-1]
            b = self.biases[-1]
            z = tf.add(tf.matmul(H, W), b)
            return z
