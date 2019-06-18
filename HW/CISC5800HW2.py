import numpy as np 
import pandas as pd

def get_sigmoid(z):
	return 1 / (1 + np.exp(-z))

# assist function 1
def get_weighted(feature, weight):
    return np.dot(feature, weight)

# assist function 2
def get_probability(feature, weight):
    return get_sigmoid(get_weighted(feature, weight))

def predict(features, weights):
	weights = pd.DataFrame(weights)
	size1 = len(features)
	size2 = weights.shape(1)
	result = []
	if size1 == size2:
		for i in size1:
			feature = features[i-1]
			weight = weights.iloc[:, i-1]
			prob = get_probability(feature, weight)
			result.append(prob)
	return result

def cost_function(features, labels, weights):
    # feature is a dataframe
    m = features.shape[0]
    total_cost = -(1 / m) * np.sum(
        label * np.log(get_probability(features, weights)) + (1 - labels) * np.log(
            1 - get_probability(features, weights)))
    return total_cost