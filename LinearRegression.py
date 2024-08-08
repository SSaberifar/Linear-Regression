import numpy as np

class LinearRegresion :
    def __init__(self , dr = 0.01 , lr = 0.01 , n_iters = 1000) :
        self.dr = dr
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, x_data , y_data) :
        
        if len(x_data.shape) == 1 :
            x_data = x_data.reshape(-1,1)
        
        n_data , n_feature  = x_data.shape
        self.weights = np.zeros(n_feature)
        self.bias = 0

        pre = 1e-5
        prev_error = 0
        for epoch in range(self.n_iters) :
            
            y_pred = np.dot(x_data , self.weights) + self.bias
            dw = np.dot(x_data.T , (y_pred - y_data)) / (n_data)
            db = np.sum(y_pred - y_data) / (n_data)
            self.lr = self.lr / ( 1 + (self.dr * epoch))

            self.weights -=  (self.lr * dw)
            self.bias -= (self.lr * db)
            
            current_error = np.mean( np.square(y_pred - y_data ))

            if abs( current_error - prev_error) < pre :
                break

            prev_error = current_error

    def predict(self, x_data) :

        if len(x_data.shape) == 1 :
            x_data = x_data.reshape(-1,1)
        return np.dot(x_data, self.weights) + self.bias