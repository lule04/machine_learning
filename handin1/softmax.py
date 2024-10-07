import numpy as np
from h1_util import numerical_grad_check

def softmax(X):
    """ 
    Compute the softmax of each row of an input matrix (2D numpy array).
    
    the numpy functions amax, log, exp, sum may come in handy as well as the keepdims=True option and the axis option.
    Remember to handle the numerical problems as discussed in the description.
    You should compute lg softmax first and then exponentiate 
    
    More precisely this is what you must do.
    
    For each row x do:
    compute max of x
    compute the log of the denominator sum for softmax but subtracting out the max i.e (log sum exp x-max) + max
    compute log of the softmax: x - logsum
    exponentiate that

    Args:
        X: numpy array shape (n, d) each row is a data point
    Returns:
        res: numpy array shape (n, d)  where each row is the softmax transformation of the corresponding row in X i.e res[i, :] = softmax(X[i, :])
    """
    res = np.zeros(X.shape)

    ### YOUR CODE HERE
    # Just follow the steps described in the description:
    max_x = np.max(X, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(X - max_x), axis=1, keepdims=True)) + max_x
    log_softmax = X - log_sum_exp
    res = np.exp(log_softmax)
    # If still 0, even after having done all the things above to avoid numerical issues,
    # then set to the smallest float in python
    res = np.where(res > 0, res, 2.2250738585072014e-308)
    ### END CODE

    return res

def one_in_k_encoding(vec, k):
    """ One-in-k encoding of vector to k classes 
    
    Args:
       vec: numpy array - data to encode
       k: int - number of classes to encode to (0,...,k-1)
    """
    n = vec.shape[0]
    enc = np.zeros((n, k))
    enc[np.arange(n), vec] = 1
    return enc
    
class SoftmaxClassifier():

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.W = None
        
    def cost_grad(self, X, y, W):
        """ 
        Compute the average negative log likelihood cost and the gradient under the softmax model 
        using data X, Y and weight matrix W.
        
        the functions np.log, np.nonzero, np.sum, np.dot (@), may come in handy
        Args:
           X: numpy array shape (n, d) float - the data each row is a data point
           y: numpy array shape (n, ) int - target values in 0,1,...,k-1
           W: numpy array shape (d x K) float - weight matrix
        Returns:
            totalcost: Average Negative Log Likelihood of w 
            gradient: The gradient of the average Negative Log Likelihood at w 
        """
        cost = np.nan
        grad = np.zeros(W.shape)*np.nan
        Yk = one_in_k_encoding(y.astype(int), self.num_classes)  # Cast y to int-array!!!
        ### YOUR CODE HERE
        # We are using the formulas given in the slides (Softmax.pdf)
        cost = -np.mean(np.log((softmax(np.matmul(X, W)) * Yk).sum(axis=1)))    # formula from slide 22/28 (with Yk)
        grad = -np.matmul(X.T, Yk-softmax(np.matmul(X, W)))/X.shape[0]   # formula from slide 23/28 (with Yk)
        ### END CODE
        return cost, grad


    def fit(self, X, Y, W=None, lr=0.01, epochs=10, batch_size=16):
        """
        Run Mini-Batch Gradient Descent on data X,Y to minimize the in sample error (1/n)NLL for softmax regression.
        Printing the performance every epoch is a good idea to see if the algorithm is working
    
        Args:
           X: numpy array shape (n, d) - the data each row is a data point
           Y: numpy array shape (n,) int - target labels numbers in {0, 1,..., k-1}
           W: numpy array shape (d x K)
           lr: scalar - initial learning rate
           batchsize: scalar - size of mini-batch
           epochs: scalar - number of iterations through the data to use

        Sets: 
           W: numpy array shape (d, K) learned weight vector matrix  W
           history: list/np.array len epochs - value of cost function after every epoch. You know for plotting
        """
        if W is None: W = np.zeros((X.shape[1], self.num_classes))
        history = []

        ### YOUR CODE HERE
        # Basically the same as in logistic regression:
        for epoch in range(epochs):
            # Permute X and y randomly
            # (Concatenation, so X_i and y_i always stay in the same line together)
            xy = np.random.permutation(np.concatenate((X, Y.reshape(Y.shape[0], 1)), axis=1))
            X = xy[:, :-1]
            Y = xy[:, -1]

            # For each batch, compute the cost and the new w
            for i in range(int(X.shape[0]/batch_size)):
                cost, g = self.cost_grad(X[i * batch_size:(i+1) * batch_size, :], Y[i * batch_size:(i+1) * batch_size], W)
                W = W - lr * g

            history.append(self.cost_grad(X, Y, W)[0])
            print("Cost after epoch", epoch+1, " :", self.cost_grad(X, Y, W)[0])

            # Reduce learning rate every 20 epochs
            if epoch % 20 == 0:
                lr = lr * 0.75
        ### END CODE
        self.W = W
        self.history = history
        

    def score(self, X, Y):
        """ Compute accuracy of classifier on data X with labels Y

        Args:
           X: numpy array shape (n, d) - the data each row is a data point
           Y: numpy array shape (n,) int - target labels numbers in {0, 1,..., k-1}
        Returns:
           out: float - mean accuracy
        """
        out = 0
        ### YOUR CODE HERE
        out = np.mean(np.equal(Y, self.predict(X)))
        ### END CODE
        return out

    def predict(self, X):
        """ Compute classifier prediction on each data point in X 

        Args:
           X: numpy array shape (n, d) - the data each row is a data point
        Returns
           out: np.array shape (n, ) - prediction on each data point (number in 0,1,..., num_classes-1)
        """
        out = None
        ### YOUR CODE HERE
        # Return the class (=index of a line) with the maximum score for each x_i
        out = np.argmax(softmax(np.matmul(X, self.W)), axis=1)
        ### END CODE
        return out


def test_encoding():
    print('*'*10, 'test encoding')
    labels = np.array([0, 2, 1, 1])
    m = one_in_k_encoding(labels, 3)
    res =  np.array([[1, 0 , 0], [0, 0, 1], [0, 1, 0], [0, 1, 0]])
    assert res.shape == m.shape, 'encoding shape mismatch'
    assert np.allclose(m, res), m - res
    print('Test Passed')

    
def test_softmax():
    print('Test softmax')
    X = np.zeros((3 ,2))
    X[0, 0] = np.log(4)
    X[1, 1] = np.log(2)
    print('Input to Softmax: \n', X)
    sm = softmax(X)
    expected = np.array([[4.0/5.0, 1.0/5.0], [1.0/3.0, 2.0/3.0], [0.5, 0.5]])
    print('Result of softmax: \n', sm)
    assert np.allclose(expected, sm), 'Expected {0} - got {1}'.format(expected, sm)
    print('Test complete')
        

def test_grad():
    print('*'*5, 'Testing  Gradient')
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]])    
    w = np.ones((2, 3))
    y = np.array([0, 1, 2])
    scl = SoftmaxClassifier(num_classes=3)
    f = lambda z: scl.cost_grad(X, y, W=z)
    numerical_grad_check(f, w)
    print('Test Success')

    
if __name__ == "__main__":
    test_encoding()
    test_softmax()
    test_grad()
