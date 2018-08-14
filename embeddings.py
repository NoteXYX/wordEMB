import numpy as np

def read(file, threshold=0, vocabulary=None, dtype='float'):
	header = file.readline().split(' ')
	count = int(header[0]) if threshold <= 0 else min(threshold, int(header[0]))
	dim = int(header[1])
	words = []
	matrix = np.empty((count,dim), dtype=dtype) if vocabulary is None else []
	for i in range(count):
		word, vec = file.readline().split(' ', 1)
		if vocabulary is None:
			words.append(word)
			matrix[i] = np.fromstring(vec, sep=' ', dtype=dtype)
		elif word in vocabulary:
			words.append(word)
			matrix.append(np.fromstring(vec, sep=' ', dtype=dtype))
	return (words, matrix) if vocabulary is None else (words, np.array(matrix, dtype=dtype))

def write(words, matrix, foutp):
	print('%d %d' % matrix.shape, file=foutp)
	for i in range(len(words)):
		print(words[i] + ' ' + ' '.join(['%.6g' % x for x in matrix[i]]), file=foutp)

def length_normalize(matrix):
	norms = np.sqrt(np.sum(matrix**2, axis=1))
	matrix /= norms[:, np.newaxis]
	return matrix

def mean_center(matrix):
	avg = np.mean(matrix, axis=0)
	matrix -= avg
	return matrix

def normalize(matrix, actions):
	for action in actions:
		if action == 'length_normalize':
			matrix = length_normalize(matrix)
		elif action == 'center':
			matrix = mean_center(matrix)
	return matrix

def propagate(w, X, Z):
	cost = np.sum((np.dot(X, w) - Z)**2)
	dw = 2 * np.dot(X.T, (np.dot(X, w) - Z))   #########################
	assert (dw.shape == w.shape)
	cost = np.squeeze(cost)
	assert (cost.shape == ())
	return dw, cost

def optimize(w, X, Z, num_iterations, learning_rate, print_cost = False):
	costs = []
	for i in range(num_iterations):
		dw, cost = propagate(w, X, Z)
		w = w - learning_rate * dw
		if i%100 == 0:
			costs.append(cost)

		if i%100 == 0 and print_cost:
			print('Cost after iteration %i: %f'% (i,cost))
	params = {'w': w,
			  'dw': dw}
	return params, costs

def predict(w, X):
	m = X.shape[0]
	dim = X.shape[1]
	Z_prediction = np.dot(X, w)
	assert (Z_prediction.shape == (m,dim))
	return Z_prediction