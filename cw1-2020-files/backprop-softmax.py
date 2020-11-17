
# Neural Computation (Extended)
# CW1: Backpropagation and Softmax
# Autumn 2020
#

import numpy as np
import time
import fnn_utils

# Some activation functions with derivatives.
# Choose which one to use by updating the variable phi in the code below.

def sigmoid(x):
    return np.array([1 / (1 + np.exp(-i)) for i in x])
def sigmoid_d(x):
    return np.array([float(sigmoid([i])*(1-sigmoid([i]))) for i in x])
def relu(x):
    return np.array([max(0,i) for i in x])
def relu_d(x):
    if type(x) is np.ndarray:
        return np.array([1 if i >0 else 0 for i in x ])
       
class BackPropagation:

    # The network shape list describes the number of units in each
    # layer of the network. The input layer has 784 units (28 x 28
    # input pixels), and 10 output units, one for each of the ten
    # classes.

    def __init__(self,network_shape=[784,20,20,20,10]):

        # Read the training and test data using the provided utility functions
        self.trainX, self.trainY, self.testX, self.testY = fnn_utils.read_data()

        # Number of layers in the network
        self.L = len(network_shape)

        self.crossings = [(1 if i < 1 else network_shape[i-1],network_shape[i]) for i in range(self.L)]

        # Create the network
        self.a             = [np.zeros(m) for m in network_shape] # output from layer
        self.db            = [np.zeros(m) for m in network_shape] # local gradiant for bias?
        self.b             = [np.random.normal(0,1/10,m) for m in network_shape] # bias
        self.z             = [np.zeros(m) for m in network_shape] # neurons
        self.delta         = [np.zeros(m) for m in network_shape] 
        self.w             = [np.random.uniform(-1/np.sqrt(m0),1/np.sqrt(m0),(m1,m0)) for (m0,m1) in self.crossings] # weights
        self.dw            = [np.zeros((m1,m0)) for (m0,m1) in self.crossings] # derivative of weights
        self.nabla_C_out   = np.zeros(network_shape[-1])

        # Choose activation function
        self.phi           = relu
        self.phi_d         = relu_d
        
        # Store activations over the batch for plotting
        self.batch_a       = [np.zeros(m) for m in network_shape]
        
        self.network_shape = network_shape
                
    def forward(self, x):
        """ Set first activation in input layer equal to the input vector x (a 24x24 picture), 
            feed forward through the layers, then return the activations of the last layer.
        """
        # TODO
        # self.a[0] = x - 0.5      # Center the input values between [-0.5,0.5]
        self.a[0] = x/255 - 0.5    # attempt to normalize
         
        for i in range(1,len(self.z)):
            self.z[i] = np.dot(self.w[i],self.a[i-1]) + self.b[i]
            if i < self.L - 1:
                self.a[i] = self.phi(self.z[i])
            else: # the softmax layer
                self.a[i] = self.softmax(self.z[i])
        np.set_printoptions(suppress=True)
        return(self.a[self.L-1])

    def softmax(self, z):
        # TODO
        q = sum([np.e**i for i in z])
        return np.array([(np.e**j)/q for j in z])

    def loss(self, pred, y):
        # TODO
        
        # Cost function 
        # Based on page: 8, slide: l06-softmax
        p_y = pred[np.argmax(y)]
        return -np.log(p_y)
    
    def backward(self,x, y):
        """ Compute local gradients, then return gradients of network.
        """
        # TODO
        
        # Local Gradients for Output layer 
        # Based on page: 10, slide: l06-softmax
        pred = self.forward(x)          # using forward function
        # pred = self.a[self.L-1]       # not using forward function (PS: if not using self.forward here, we need to add it in self.sgd )
        self.delta[self.L-1] = pred - y
        
        # Local Gradients for Hidden layer l
        # Based on page: 15, slide: l05-backpropagation
        for i in range(len(self.delta)-2, -1, -1):
            # print(self.delta[i])
            # print(i)
            self.delta[i] = self.phi_d(self.z[i]) * (self.w[i+1].T @ self.delta[i+1])

        # Partial derivatives
        # Based on page: 16, slide: l05-backpropagation
        for i in range(1, self.L):
            self.dw[i] = np.asmatrix(self.delta[i]).T @ np.asmatrix(self.a[i-1])
            self.db[i] = self.delta[i]
        

    # Return predicted image class for input x
    def predict(self, x):
        # return # TODO
        # Just go through the forward function to make a prediction
        return np.argmax(self.forward(x))

    # Return predicted percentage for class j
    def predict_pct(self, j):
        # return # TODO 
        return self.a[self.L-1][j]
    
    def evaluate(self, X, Y, N):
        """ Evaluate the network on a random subset of size N. """
        num_data = min(len(X),len(Y))
        samples = np.random.randint(num_data,size=N)
        results = [(self.predict(x), np.argmax(y)) for (x,y) in zip(X[samples],Y[samples])]
        return sum(int(x==y) for (x,y) in results)/N

    
    def sgd(self,
            batch_size=50,
            epsilon=0.01,
            # epochs=1000):
            epochs=10):      # I think 5-10 is enough to quickly check the performance of the model. However, we can adjust it for task 6

        """ Mini-batch gradient descent on training data.

            batch_size: number of training examples between each weight update
            epsilon:    learning rate
            epochs:     the number of times to go through the entire training data
        """
        
        # Compute the number of training examples and number of mini-batches.
        N = min(len(self.trainX), len(self.trainY))
        num_batches = int(N/batch_size)

        # Variables to keep track of statistics
        loss_log      = []
        test_acc_log  = []
        train_acc_log = []

        timestamp = time.time()
        timestamp2 = time.time()

        predictions_not_shown = True
        
        # In each "epoch", the network is exposed to the entire training set.
        for t in range(epochs):

            # We will order the training data using a random permutation.
            permutation = np.random.permutation(N)
            
            # Evaluate the accuracy on 1000 samples from the training and test data
            test_acc_log.append( self.evaluate(self.testX, self.testY, 1000) )
            train_acc_log.append( self.evaluate(self.trainX, self.trainY, 1000))
            batch_loss = 0

            for k in range(num_batches):
                
                # Reset buffer containing updates
                # TODO
                # Make empty containers to store the sum value of partial derivatives in one batch. 
                # Based on page: 18, slide: l05-backpropagation
                dw_buffer = [i*0 for i in self.dw]
                db_buffer = [i*0 for i in self.db]
                
                # Mini-batch loop
                for i in range(batch_size):

                    # Select the next training example (x,y)
                    x = self.trainX[permutation[k*batch_size+i]]
                    y = self.trainY[permutation[k*batch_size+i]]

                    # Feed forward inputs
                    # TODO
                    self.backward(x, y)     # self.forward is included in self.backward
                    
                    # Compute gradients
                    # TODO
                    # To sum the partial derivatives for each parameter. 
                    # Based on page: 18, slide: l05-backpropagation
                    for l in range(self.L):
                        dw_buffer[l] += self.dw[l]
                        db_buffer[l] += self.db[l]

                    # Update loss log
                    batch_loss += self.loss(self.a[self.L-1], y)

                    for l in range(self.L):
                        self.batch_a[l] += self.a[l] / batch_size
                                    
                # Update the weights at the end of the mini-batch using gradient descent
                for l in range(1,self.L):
                    # Based on page: 18, slide: l05-backpropagation
                    # self.w[l] = # TODO
                    self.w[l] -= epsilon * (dw_buffer[l] / batch_size)
                    # self.b[l] = # TODO
                    self.b[l] -= epsilon * (db_buffer[l] / batch_size)
                
                # Update logs
                loss_log.append( batch_loss / batch_size )
                batch_loss = 0

                # Update plot of statistics every 10 seconds.
                if time.time() - timestamp > 10:
                    timestamp = time.time()
                    fnn_utils.plot_stats(self.batch_a,
                                         loss_log,
                                         test_acc_log,
                                         train_acc_log)

                # Display predictions every 20 seconds.
                if (time.time() - timestamp2 > 20) or predictions_not_shown:
                    predictions_not_shown = False
                    timestamp2 = time.time()
                    fnn_utils.display_predictions(self,show_pct=True)

                # Reset batch average
                for l in range(self.L):
                    self.batch_a[l].fill(0.0)
        
        fnn_utils.save_pic(epochs, epsilon, batch_size, self.network_shape)
        return test_acc_log, train_acc_log


# Start training with default parameters.

def main():
    start_time = time.time()
    
    # Find a best network topology
    best_network_shape = find_topology()
    
    # Find a best learning rate
    # best_epsilon = find_epsilon(epoch=best_epoch)
    # best_epsilon = find_epsilon()
    
    # Find a best minibatch size
    # best_batch_size = find_batch_size(epoch=best_epoch, epsilon=best_epsilon)
    # best_batch_size = find_batch_size()
    
    # Find a best epoch
    # best_epoch = find_epoch()
    
    # Final test
    bp = BackPropagation(
                            # network_shape = best_network_shape
                            )
    test_acc_log, train_acc_log = bp.sgd(
                                            # batch_size=best_batch_size, 
                                            # epsilon=best_epsilon, 
                                            # epochs=best_epoch,
                                            )
    
    end_time = time.time()
    
    print_msg(test_acc_log, train_acc_log, int(end_time - start_time))  
    
    input("Press enter to continue...")     # To keep the program from shutting down, so that we can see the final result. We can delete it later.
   

def find_topology(epochs=15, epsilon=0.1, batch_size=32):
    # network_shape_pool = [[784,20,20,20,10]]
    network_shape_pool = [
                            # layer 6
##                            [784,10,10,10,10,10],
##                            [784,20,20,20,20,10],
##                            [784,30,30,30,30,10],
##                            [784,40,40,40,40,10],
##                            [784,50,50,50,50,10],
##                            [784,60,60,60,60,10],
##                            [784,70,70,70,70,10],
##                            [784,80,80,80,80,10],
##                            [784,90,90,90,90,10],
                            # layer 7
                            [784,10,10,10,10,10,10],
                            [784,20,20,20,20,20,10],
                            [784,30,30,30,30,30,10],
                            [784,40,40,40,40,40,10],
                            [784,50,50,50,50,50,10],
                            [784,60,60,60,60,60,10],
                            [784,70,70,70,70,70,10],
                            [784,80,80,80,80,80,10],
                            [784,90,90,90,90,90,10]]
    final_accuracy = list()
    average_accuracy = list()
    for network_shape in network_shape_pool:
        start_time = time.time()
        bp = BackPropagation(network_shape=network_shape)
        test_acc_log, train_acc_log = bp.sgd(epochs=epochs, epsilon=epsilon, batch_size=batch_size)
        end_time = time.time()
        
        final_accuracy.append(test_acc_log[-1]) 
        average_accuracy.append(round(np.mean(test_acc_log[1:]),3))
        
        print_msg(test_acc_log, train_acc_log, int(end_time - start_time), epsilon=epsilon, batch_size=batch_size, network_shape=network_shape) 
        del bp
        
    # best_network_shape = network_shape_pool[np.argmax(final_accuracy)]
    best_network_shape = network_shape_pool[np.argmax(average_accuracy)]
    print("network_shape_pool:", network_shape_pool)
    print("final_accuracy:", final_accuracy)
    print("average_accuracy:", average_accuracy)
    print("best_network_shape:", best_network_shape)
    return best_network_shape 


def find_epoch(epsilon=0.01, batch_size=50):
    # epoch_pool = list(range(25,30))
    epoch_pool = [50,100,150,200]
    best_epoch_list = list()
    for epoch in epoch_pool:
        start_time = time.time()
        bp = BackPropagation()
        test_acc_log, train_acc_log = bp.sgd(epochs=epoch, epsilon=epsilon, batch_size=batch_size)
        end_time = time.time()
        
        best_epoch = np.argmax(test_acc_log) + 1
        best_epoch_list.append(best_epoch)
        
        print_msg(test_acc_log, train_acc_log, int(end_time - start_time), epsilon=epsilon, batch_size=batch_size)     
        
    from collections import Counter
    best_epoch = Counter(best_epoch_list).most_common(1)[0][0]
    print("epoch_pool:", epoch_pool)
    print("best_epoch_list:", best_epoch_list)
    print("best_epoch:", best_epoch)
    return best_epoch


def find_epsilon(epoch=15, batch_size=50):
    epsilon_pool = [0.001,0.003,0.005,0.007,0.009,0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.4,0.6,0.8]
    final_accuracy = list()
    average_accuracy = list()
    for epsilon in epsilon_pool:
        start_time = time.time()
        bp = BackPropagation()
        test_acc_log, train_acc_log = bp.sgd(epochs=epoch, epsilon=epsilon, batch_size=batch_size)
        end_time = time.time()
        
        final_accuracy.append(test_acc_log[-1]) 
        average_accuracy.append(round(np.mean(test_acc_log[1:]),3))
        
        print_msg(test_acc_log, train_acc_log, int(end_time - start_time), epsilon=epsilon, batch_size=batch_size)  
    
    best_epsilon = epsilon_pool[np.argmax(average_accuracy)]
    print("epsilon_pool:", epsilon_pool)
    print("final_accuracy:", final_accuracy)
    print("average_accuracy:", average_accuracy)
    print("best_epsilon:", best_epsilon)
    return best_epsilon    
   
   
def find_batch_size(epoch=15, epsilon=0.01):
    batch_size_pool = [1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,200]
    final_accuracy = list()
    average_accuracy = list()
    for batch_size in batch_size_pool:
        start_time = time.time()
        bp = BackPropagation()
        test_acc_log, train_acc_log = bp.sgd(epochs=epoch, epsilon=epsilon, batch_size=batch_size)
        end_time = time.time()
        
        final_accuracy.append(test_acc_log[-1]) 
        average_accuracy.append(round(np.mean(test_acc_log[1:]),3))
        
        print_msg(test_acc_log, train_acc_log, int(end_time - start_time), epsilon=epsilon, batch_size=batch_size)  
    
    best_batch_size = batch_size_pool[np.argmax(average_accuracy)]
    print("batch_size_pool:", batch_size_pool)
    print("final_accuracy:", final_accuracy)
    print("average_accuracy:", average_accuracy)
    print("best_batch_size:", best_batch_size)
    return best_batch_size 


def print_msg(test_acc_log, train_acc_log, time_consumption=str(), epsilon=None, batch_size=None, network_shape=None):
    print(str())
    if network_shape:
        print("Network_shape:", network_shape)
    if epsilon:
        print("Epsilon:", epsilon)
    if batch_size:
        print("Batch_size:", batch_size)
    print("Epochs:", len(test_acc_log))
    print("train_acc_log:", train_acc_log)
    print("test_acc_log:", test_acc_log)
    print("The Highest Accuracy is %s in epoch: %s" % (max(test_acc_log), np.argmax(test_acc_log) + 1))
    print("Time consumption:", time_consumption, "seconds")
    print("The Final Accuracy:", test_acc_log[-1])
    print("The Average Accuracy:", round(np.mean(test_acc_log[1:]),3))
    print(str())


if __name__ == "__main__":
    main()
    
