import numpy as np 
import matplotlib.pyplot as plt 
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


np.random.seed(1)

#Dataset 1

X, Y = load_planar_dataset()

#Dataset 2
'''noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}

### START CODE HERE ### (choose your dataset)
dataset = "noisy_moons"
### END CODE HERE ###

X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])

# make blobs binary
if dataset == "blobs":
    Y = Y%2

# Visualize the data
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);'''

#print(X.shape) #(2, 400) 
#print(Y.shape) #(1, 400) 
#print(X.shape[1]) #Number of training examples = 400

#Visualize the data:

#plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
#plt.show()

#Logistic Regression

'''clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T,Y.T)

plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")

LR_prediction = clf.predict(X.T)
print(clf.score)
plt.show()
print('Accuracy of logistic regression: %d' % float((np.dot(Y,LR_prediction) + np.dot(1-Y,1-LR_prediction))/float(Y.size)*100) + '%')'''

#Layer Sizes

def layer_sizes(X, Y):
	n_x = len(X) # Number of features = Row size
	n_h = 4
	n_y = len(Y)

	return (n_x,n_h,n_y)

#Initialising paramters

def initialize_parameters(n_x,n_h,n_y):
	W1 = np.random.randn(n_h,n_x) * 0.01
	b1 = np.zeros((n_h,1))
	W2 = np.random.randn(n_y,n_h) * 0.01
	b2 = np.zeros((n_y,1))

	parameters = {"W1":W1,
				  "b1":b1,
				  "W2":W2,
				  "b2":b2}

	return parameters

#Forward propagation

def forward_propagation(X, parameters):

	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]

	Z1 = np.dot(W1,X) + b1
	A1 = np.tanh(Z1)
	Z2 = np.dot(W2,A1) + b2
	A2 = sigmoid(Z2)

	cache = {"Z1":Z1,
			 "A1":A1,
			 "Z2":Z2,
			 "A2":A2}

	return A2, cache


def compute_cost(A2, Y, parameters):

	m = Y.shape[1]

	logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),(1-Y))
	cost = (-1/m) * np.sum(logprobs)

	cost = np.squeeze(cost) # turns [[10]] into 10 

	return cost

'''A2, Y_assess, parameters = compute_cost_test_case()
print("cost = " + str(compute_cost(A2, Y_assess, parameters)))'''

#Backward propagation

def backward_propagation(parameters, cache, X, Y):

	m = X.shape[1]

	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]


	Z1 = cache["Z1"]
	A1 = cache["A1"]
	Z2 = cache["Z2"]
	A2 = cache["A2"]

	dZ2 = A2 - Y
	dW2 = (1/m) * np.dot(dZ2,A1.T)
	db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
	dZ1 = np.dot(W2.T,dZ2) * (1-np.power(A1,2))
	dW1 = (1/m) * np.dot(dZ1,X.T)
	db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

	grads = {"dW1":dW1,
			 "db1":db1,
			 "dW2":dW2,
			 "db2":db2}

	return grads

'''parameters, cache, X_assess, Y_assess = backward_propagation_test_case()

grads = backward_propagation(parameters, cache, X_assess, Y_assess)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("db2 = "+ str(grads["db2"]))'''


def update_parameters(parameters, grads, learning_rate=1.2):

	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]

	dW1 = grads["dW1"]
	db1 = grads["db1"]
	dW2 = grads["dW2"]
	db2 = grads["db2"]

	W1 = W1 - learning_rate * dW1
	b1 = b1 - learning_rate * db1
	W2 = W2 - learning_rate * dW2
	b2 = b2 - learning_rate * db2


	parameters = {"W1":W1,
				  "b1":b1,
				  "W2":W2,
				  "b2":b2}

	return parameters

'''parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))'''


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):

	np.random.seed(3)
	n_x = layer_sizes(X,Y)[0]
	n_y = layer_sizes(X,Y)[2]


	parameters = initialize_parameters(n_x,n_h,n_y)

	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]



	for i in range(0, num_iterations):

		A2, cache = forward_propagation(X, parameters)

		cost = compute_cost(A2, Y, parameters)

		grads = backward_propagation(parameters, cache, X, Y)

		parameters = update_parameters(parameters, grads, learning_rate=1.2)

		if print_cost and i % 1000 == 0:
        		print ("Cost after iteration %i: %f" %(i, cost))

	return parameters


'''X_assess, Y_assess = nn_model_test_case()
parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=True)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))'''

def predict(parameters, X):

	A2, cache = forward_propagation(X, parameters)

	predictions = A2 > 0.5

	return predictions


'''parameters, X_assess = predict_test_case()

predictions = predict(parameters, X_assess)
print("predictions mean = " + str(np.mean(predictions)))'''

parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()

predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')















