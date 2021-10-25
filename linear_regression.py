import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATA_NAME = 'ex1data1.txt'

class LinearRegression:
    def __init__(self, data):
        self.data = data
        self.x_train, self.y_train = data[0], data[1]
        self.x_train_modified = np.asarray([[1, i] for i in self.x_train], dtype=np.float32)
        self.y_train_modified = np.asarray([[i] for i in self.y_train], dtype=np.float32)
        self.data_size = len(self.y_train)
        self.alpha = 0.01
        self.iterations = 1500
        self.cost_history = np.zeros((self.iterations, 1))

    def plot_data(self):
        plt.plot(self.x_train, self.y_train, 'rx')
        plt.ylabel('Profit in $10,000s')
        plt.xlabel('Population of City in 10,000s')
        plt.show()
    
    def cost_computation(self, weights: np.ndarray = np.zeros((2, 1))) -> int:
        cost = 0
        prediction = np.dot(self.x_train_modified, weights) # h = wX
        square_error = np.square(prediction - self.y_train_modified) # square_error = (prediction - self.y_train_modified)**2 # square error = (h - y)^2
        cost = 1/(2*self.data_size) * np.sum(square_error) # j = (1/2m) * sum(square error)
        return cost
    
    def gradient_descent(self, weights: np.ndarray = np.zeros((2, 1))):
        for i in range(self.iterations):
            prediction = np.dot(self.x_train_modified, weights) # h = wX
            error = prediction - self.y_train_modified # error = h - y
            gradient = np.matrix(self.alpha * (1/self.data_size) * np.multiply(self.x_train_modified, error).sum(axis=0)).transpose() # gradient = alpha*(1/m)*sum(y*error)^T
            weights = weights - gradient # w = w - gradient
            self.cost_history[i][0] = self.cost_computation(weights)
        return weights
    
    def plot_cost_history(self, weights: np.ndarray):
        a = weights.item(1)
        b = weights.item(0)
        y_list = []
        x_list = np.arange(5.0, 22.5, 0.5)
        for i in x_list:
            y_list.append((a*i)+b)
        plt.plot(self.x_train, self.y_train, 'rx')
        plt.plot(x_list, y_list)
        plt.ylabel('Profit in $10,000s')
        plt.xlabel('Population of City in 10,000s')
        plt.legend(['Training data', 'Linear regression'])
        plt.show()
        
    def predict(self, weights: np.ndarray, population_size: int) -> int:
        a = weights.item(1)
        b = weights.item(0)
        x = population_size/10000
        return int(((a*x)+b)*10000)

if __name__=="__main__":
    data = pd.read_csv(DATA_NAME, header=None)
    linear_regression = LinearRegression(data)
    linear_regression.plot_data()
    print(f'test cost coputation: {linear_regression.cost_computation(weights=np.asarray([[-1], [2]], dtype=np.float32))}')
    weights = linear_regression.gradient_descent()
    print(f'weights: {weights}')
    linear_regression.plot_cost_history(weights)
    print(f'for population=35000, we predict a profit of {linear_regression.predict(weights, population_size=35000)}')
    print(f'for population=70000, we predict a profit of {linear_regression.predict(weights, population_size=70000)}')
