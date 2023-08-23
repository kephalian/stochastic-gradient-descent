# use a distribution to generate data points for the stochastic gradient descent visualization. Here's an example using a normal distribution to generate data points:

#```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data from a normal distribution
np.random.seed(42)
X = np.random.randn(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Stochastic Gradient Descent parameters
eta = 0.01  # learning rate
n_iterations = 1000
m = len(X)

# Initialize the parameters for the linear model
theta = np.random.randn(2, 1)


#Certainly! This code snippet performs Stochastic Gradient Descent (SGD) to optimize the parameters of a linear regression model. Let's break down the code step by step:

#1. `for iteration in range(n_iterations):` - This loop iterates over a specified number of iterations (`n_iterations`) to perform the stochastic gradient descent algorithm.

#2. `random_index = np.random.randint(m)` - Generates a random index from the range of available data points (`m` is the total number of data points). This index will be used to select a random data point for each iteration of SGD.

#3. `xi = np.c_[np.ones((1, 1)), X[random_index:random_index + 1]]` - Prepares the input data `xi` for the selected random data point. It adds a bias term (constant 1) to the input feature and selects the random data point at the `random_index`. This is done to match the dimensions for matrix operations.

#4. `yi = y[random_index:random_index + 1]` - Selects the corresponding target output (`yi`) for the selected random data point.

#5. `gradients = 2 * xi.T.dot(xi.dot(theta) - yi)` - Computes the gradients of the cost function with respect to the parameters `theta`. The gradients are calculated using the derivative of the Mean Squared Error (MSE) loss function with respect to `theta`. This step is an essential part of gradient descent, and the stochastic nature comes from the fact that we're using a single data point for each iteration.

#6. `theta = theta - eta * gradients` - Updates the parameters `theta` of the linear regression model using the calculated gradients. `eta` is the learning rate, which controls the step size of the parameter updates. The parameter updates are subtracted from the current values of `theta`.

#The loop iterates over the specified number of iterations, and in each iteration, a random data point is selected, gradients are computed based on that data point, and the model parameters are updated accordingly. Over time, this process aims to minimize the cost function and find the best-fitting linear regression line for the given data.

#The end result is a linear regression line that fits the data points more effectively, achieved by iteratively adjusting the model parameters using stochastic gradient descent.


# Perform Stochastic Gradient Descent
for iteration in range(n_iterations):
    random_index = np.random.randint(m)
    xi = np.c_[np.ones((1, 1)), X[random_index:random_index + 1]]
    yi = y[random_index:random_index + 1]
    gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
    theta = theta - eta * gradients

# Plot the generated data points
plt.scatter(X, y, label='Generated Data Points')

# Plot the linear regression line
x_range = np.linspace(-3, 3, 100).reshape(-1, 1)
x_range_with_bias = np.c_[np.ones((100, 1)), x_range]
y_range = x_range_with_bias.dot(theta)
plt.plot(x_range, y_range, color='red', label=f'Stochastic Gradient Descent Learning rate {eta} and Iterations {n_iterations}')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Stochastic Gradient Descent with Randomized Normal Distribution Data')
plt.legend()
plt.savefig('tt.png', dpi=400)
plt.show()
#```

#In this example, the `np.random.randn` function is used to generate data points from a normal distribution. Adjust the parameters and distribution as needed for your visualization.
