import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class LinearRegression:
    def __init__(self, data):
        self.data = data
        self.x_train = self.data.iloc[: , :-1]
        self.y_train = self.data.iloc[: , -1]
        self.data_size = len(self.y_train)
        self.x_train_modified = np.asarray(self.x_train, dtype=np.float32)
        self.x_train_modified = np.c_[np.ones(self.data_size), self.x_train_modified] # add one column at start
        self.y_train_modified = np.asarray([[i] for i in self.y_train], dtype=np.float32)
        self.alpha = 0.01
        self.iterations = 1500
        self.cost_history = np.zeros((self.iterations, 1))
    
    def feature_normalization(self):
        temp_data = self.x_train_modified[:,1:] # we don't need first column to be normalize
        mu = temp_data.mean(axis=0)
        sigma = temp_data.std(axis=0)
        normalized_features = (temp_data-mu)/sigma
        self.x_train_modified = np.c_[np.ones(self.data_size), normalized_features] # add one column at start
    
    def plot_data(self, feature_num: int = 0, ylabel: str = '', xlabel: str = ''):
        plt.plot(self.x_train[feature_num], self.y_train, 'rx')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.show()
    
    def cost_computation(self, weights: np.ndarray) -> int:
        cost = 0
        prediction = np.dot(self.x_train_modified, weights) # h = wX
        square_error = np.square(prediction - self.y_train_modified) # square_error = (prediction - self.y_train_modified)**2 # square error = (h - y)^2
        cost = 1/(2*self.data_size) * np.sum(square_error) # j = (1/2m) * sum(square error)
        return cost
    
    def gradient_descent(self, weights: np.ndarray):
        for i in range(self.iterations):
            prediction = np.dot(self.x_train_modified, weights) # h = wX
            error = prediction - self.y_train_modified # error = h - y
            gradient = np.matrix(self.alpha * (1/self.data_size) * np.multiply(self.x_train_modified, error).sum(axis=0)).transpose() # gradient = alpha*(1/m)*sum(y*error)^T
            weights = weights - gradient # w = w - gradient
            self.cost_history[i][0] = self.cost_computation(weights)
        return weights
    
    def plot_cost_history(self):
        plt.plot(self.cost_history)
        plt.ylabel('Error')
        plt.xlabel('Iteration')
        plt.show()
       
    def normal_equation(self):
        weights = np.dot(np.dot(np.linalg.inv(np.dot(self.x_train_modified.transpose(), self.x_train_modified)), self.x_train_modified.transpose()), self.y_train_modified)
        return weights
    
    def plot_linear_regression(self, weights: np.ndarray, feature_num: int = 0, ylabel: str = '', xlabel: str = ''):
        a = weights.item(1)
        b = weights.item(0)
        y_list = []
        x_list = np.arange(5.0, 22.5, 0.5)
        for i in x_list:
            y_list.append((a*i)+b)
        plt.plot(self.x_train[feature_num], self.y_train, 'rx')
        plt.plot(x_list, y_list)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend(['Training data', 'Linear regression'])
        plt.show()
        
    def predict(self, weights: np.ndarray, X: np.ndarray) -> int:
        X = np.asarray(X, dtype=np.float32)
        return int(np.dot(X, weights)[0])

class LinearRegressionTest:
    def single_feature_linear_regression():
        SINGLE_FEATURE_DATA_NAME = 'ex1data1.txt'
        data = pd.read_csv(SINGLE_FEATURE_DATA_NAME, header=None)
        linear_regression = LinearRegression(data)
        linear_regression.plot_data(ylabel='Profit in $10,000s', xlabel='Population of City in 10,000s')
        print(f'test cost coputation: {linear_regression.cost_computation(weights=np.asarray([[-1], [2]], dtype=np.float32))}')
        features_count = data.shape[1]
        weights = linear_regression.gradient_descent(weights = np.zeros((features_count, 1)))
        print(f'weights: {weights}')
        linear_regression.plot_cost_history()
        linear_regression.plot_linear_regression(weights, ylabel='Profit in $10,000s', xlabel='Population of City in 10,000s')
        print(f'for population=35000, we predict a profit of {linear_regression.predict(weights, X=[1, 35000]):,}')
        print(f'for population=70000, we predict a profit of {linear_regression.predict(weights, X=[1, 70000]):,}')
        
    def multi_feature_linear_regression():
        MULTI_FEATURE_DATA_NAME = 'ex1data2.txt'
        data = pd.read_csv(MULTI_FEATURE_DATA_NAME, header=None)
        linear_regression = LinearRegression(data)
        linear_regression.feature_normalization()
        linear_regression.plot_data(feature_num=0, ylabel='House Price', xlabel='Area(sq-ft)')
        features_count = data.shape[1]
        weights = linear_regression.gradient_descent(weights = np.zeros((features_count, 1)))
        print(f'weights: {weights}')
        linear_regression.plot_cost_history()
        linear_regression.plot_linear_regression(weights, feature_num=0, ylabel='House Price', xlabel='Area(sq-ft)')
        print(f'for area=1650 and 3 bedrooms, we predict a price of {linear_regression.predict(weights, X=[1, 1650, 3]):,}')
        
    def normal_equation_linear_regression():
        MULTI_FEATURE_DATA_NAME = 'ex1data2.txt'
        data = pd.read_csv(MULTI_FEATURE_DATA_NAME, header=None)
        linear_regression = LinearRegression(data)
        linear_regression.plot_data(feature_num=0, ylabel='House Price', xlabel='Area(sq-ft)')
        weights = linear_regression.normal_equation()
        print(f'weights: {weights}')
        linear_regression.plot_linear_regression(weights, feature_num=0, ylabel='House Price', xlabel='Area(sq-ft)')
        print(f'for area=1650 and 3 bedrooms, we predict a price of {linear_regression.predict(weights, X=[1, 1650, 3]):,}')
        
if __name__=="__main__":
    # LinearRegressionTest.single_feature_linear_regression()
    # LinearRegressionTest.multi_feature_linear_regression()
    LinearRegressionTest.normal_equation_linear_regression()
