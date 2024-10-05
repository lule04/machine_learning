import numpy as np
from h1_util import numerical_grad_check


def logistic(z):
    """ 
    Helper function
    Computes the logistic function 1/(1+e^{-x}) to each entry in input vector z.
    
    np.exp may come in handy
    Args:
        z: numpy array shape (d,) 
    Returns:
       logi: numpy array shape (d,) each entry transformed by the logistic function 
    """
    logi = np.zeros(z.shape)

    ### YOUR CODE HERE
    logi = np.where(1/(1+np.exp(-z)) > 0, 1/(1+np.exp(-z)), 2.2250738585072014e-308)
    # To reduce numerical issues: Use the smallest float in python, if 1/(1+np.exp(-z)) is evaluated to 0
    ### END CODE

    assert logi.shape == z.shape
    return logi


class LogisticRegressionClassifier():

    def __init__(self):
        self.w = None

    def cost_grad(self, X, y, w):
        """
        Compute the average negative log likelihood and gradient under the logistic regression model 
        using data X, targets y, weight vector w 
        
        np.log, np.sum, np.choose, np.dot may be useful here
        Args:
           X: np.array shape (n,d) float - Features 
           y: np.array shape (n,)  int - Labels 
           w: np.array shape (d,)  float - Initial parameter vector

        Returns:
           cost: scalar: the average negative log likelihood for logistic regression with data X, y 
           grad: np.array shape(d, ) gradient of the average negative log likelihood at w 
        """
        cost = 0
        grad = np.zeros(w.shape)

        ### YOUR CODE HERE
        # We are using the formulas given in the slides (GradientDescent.pdf)
        cost = np.mean(-np.log(logistic(y * np.matmul(X, w))))   # formula from slide 5/40
        grad = np.matmul(X.T, -y * logistic(- y * np.matmul(X, w)))/y.shape[0]  # formula from slide 11/40
        ### END CODE

        assert grad.shape == w.shape
        return cost, grad

    def fit(self, X, y, w=None, lr=0.1, batch_size=16, epochs=50):
        """
        Run mini-batch stochastic Gradient Descent for logistic regression 
        use batch_size data points to compute gradient in each step.
    
        The function np.random.permutation may prove useful for shuffling the data before each epoch
        It is wise to print the performance of your algorithm at least after every epoch to see if progress is being made.
        Remember the stochastic nature of the algorithm may give fluctuations in the cost as iterations increase.

        Args:
           X: np.array shape (n,d) dtype float32 - Features 
           y: np.array shape (n,) dtype int32 - Labels 
           w: np.array shape (d,) dtype float32 - Initial parameter vector
           lr: scalar - learning rate for gradient descent
           batch_size: number of elements to use in minibatch
           epochs: Number of scans through the data

        sets: 
           w: numpy array shape (d,) learned weight vector w
           history: list/np.array len epochs - value of loss function (in-sample error) after every epoch. Used for plotting
        """
        if w is None: w = np.zeros(X.shape[1])
        history = []

        ### YOUR CODE HERE
        for epoch in range(epochs):
            # Permute X and y randomly
            # (Concatenation, so X_i and y_i always stay in the same line together)
            xy = np.random.permutation(np.concatenate((X, y.reshape(y.shape[0], 1)), axis=1))
            X = xy[:, :-1]
            y = xy[:, -1]

            # For each batch, compute the cost and the new w
            for i in range(int(X.shape[0]/batch_size)):
                cost, g = self.cost_grad(X[i * batch_size:(i+1) * batch_size, :], y[i * batch_size:(i+1) * batch_size], w)
                w = w - lr * g
            history.append(self.cost_grad(X, y, w)[0])
            print("Cost after epoch", epoch+1, " :", self.cost_grad(X, y, w)[0])

            # Reduce learning rate every 20 epochs
            if epoch % 20 == 0:
                lr = lr * 0.75
        ### END CODE

        self.w = w
        self.history = history

    def predict(self, X):
        """ Classify each data element in X.

        Args:
            X: np.array shape (n,d) dtype float - Features 
        
        Returns: 
           p: numpy array shape (n, ) dtype int32, class predictions on X (-1, 1). NOTE: We want a class here, 
           not a probability between 0 and 1. You should thus return the most likely class!

        """
        out = np.ones(X.shape[0])

        ### YOUR CODE HERE
        out = np.array([1 if i > 0.5 else -1 for i in logistic(np.matmul(X, self.w))])
        ### END CODE

        return out
    
    def score(self, X, y):
        """ Compute model accuracy on Data X with labels y

        Args:
            X: np.array shape (n,d) dtype float - Features 
            y: np.array shape (n,) dtype int - Labels 

        Returns: 
           s: float, number of correct predictions divided by n. NOTE: This is accuracy, not in-sample error!

        """
        s = 0

        ### YOUR CODE HERE
        s = np.mean(np.isclose(y, self.predict(X), rtol=1e-08, atol=0.0))
        # Using the np.isclose() function to reduce the errors caused by the numerical issues
        ### END CODE

        return s
        

def test_logistic():
    print('*'*5, 'Testing logistic function')
    a = np.array([0, 1, 2, 3])
    lg = logistic(a)
    target = np.array([ 0.5, 0.73105858, 0.88079708, 0.95257413])
    assert np.allclose(lg, target), 'Logistic Mismatch Expected {0} - Got {1}'.format(target, lg)
    print('Test Success!')

    
def test_cost():
    print('*'*5, 'Testing Cost Function')
    X = np.array([[1.0, 0.0], [1.0, 1.0], [3, 2]])
    y = np.array([-1, -1, 1], dtype='int64')
    w = np.array([0.0, 0.0])
    print('shapes', X.shape, w.shape, y.shape)
    lr = LogisticRegressionClassifier()
    cost,_ = lr.cost_grad(X, y, w)
    target = -np.log(0.5)
    assert np.allclose(cost, target), 'Cost Function Error:  Expected {0} - Got {1}'.format(target, cost)
    print('Test Success')

    
def test_grad():
    print('*'*5, 'Testing  Gradient')
    X = np.array([[1.0, 0.0], [1.0, 1.0], [2.0, 3.0]])    
    w = np.array([0.0, 0.0])
    y = np.array([-1, -1, 1]).astype('int64')
    print('shapes', X.shape, w.shape, y.shape)
    lr = LogisticRegressionClassifier()
    f = lambda z: lr.cost_grad(X, y, w=z)
    numerical_grad_check(f, w)
    print('Test Success')

    
if __name__ == '__main__':
    test_logistic()
    test_cost()
    test_grad()
    
    
