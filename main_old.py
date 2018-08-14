from gensim.models import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orth


def initialize(dim_D, dim_W):
	dictionary = np.zeros((dim_D, dim_D))
	for i in range(dim_D):
		dictionary[i][i] = 1
	W = np.zeros((dim_W, dim_W))
	#W = np.random.randn(dim_W, dim_W)
	return dictionary, W

def normalize(X):
	mean_X = np.mean(X, axis=0)
	std_X = np.std(X, axis=0)
	X = (X - mean_X) / std_X
	return X

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

def train(model_source, model_target, vocab_array, num_iterations=2000, learning_rate=0.05, print_cost = False):
	vec_EN = np.zeros((1, 300))
	vec_IT = np.zeros((1, 300))
	D, w = initialize(25, 300)
	true_word = 0.0
	m = vocab_array.shape[0]
	for i in range(m):
		word_EN = vocab_array[i][0]
		row_EN = model_source.wv[word_EN]
		row_EN.shape = (1, 300)
		vec_EN = np.row_stack((vec_EN, row_EN))
		word_IT = vocab_array[i][1]
		row_IT = model_target.wv[word_IT]
		row_IT.shape = (1, 300)
		vec_IT = np.row_stack((vec_IT, row_IT))
	vec_EN = np.delete(vec_EN, 0, 0)	#25*300
	vec_IT = np.delete(vec_IT, 0, 0)	#25*300
	params, costs = optimize(w, vec_EN, vec_IT, num_iterations, learning_rate, print_cost)
	w = params['w']
	Z_prediction = predict(w, vec_EN)
	for i in range(Z_prediction.shape[0]):
		vec_prediction = Z_prediction[i]
		vec_prediction.shape = (300, )
		e = model_target.wv.similar_by_vector(vec_prediction, topn=1, restrict_vocab=None)
		#print(e[0][0])
		if e[0][0] == vocab_array[i][1]:
			true_word += 1

	print('Train accuracy: {}%'.format(true_word/m*100))
	d = {'costs': costs,
		 'w': w,
		 'learning_rate': learning_rate}
	return d







if __name__ == '__main__':
	model_EN = KeyedVectors.load_word2vec_format('model_EN/v7_EN_nor.vec', binary=False)
	print('英文模型加载完毕！')
	model_IT = KeyedVectors.load_word2vec_format('model_ES/v7_ES_nor.vec', binary=False)
	print('意大利文模型加载完毕！')
	#model_EN = Word2Vec.load("model_EN/v7_EN.model")
	#model_IT = Word2Vec.load("model_ES/v7_ES.model")
	vocab = np.load('vocab/vocabEN-ES.npy')
	vocab_train = np.array([['','']])
	for i in range(100):
		row = np.array([vocab[i][0], vocab[i][1]])
		vocab_train = np.row_stack((vocab_train, row))
	vocab_train = np.delete(vocab_train, 0, 0)
	print(vocab_train)
	learning_rates = [0.001,0.0001]
	models = {}
	for i in learning_rates:
		print('learn_rate is :' + str(i))
		models[str(i)] = train(model_EN, model_IT, vocab_train, num_iterations=1000, learning_rate=i, print_cost=False)
		print('\n' + '---------------------------------------------------------------------' + '\n')

	for i in learning_rates:
		plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))
	#d = train(model_EN, model_IT, vocab_train, num_iterations=3000, learning_rate=0.001, print_cost=True)
	plt.ylabel('cost')
	plt.xlabel('iterations (per hundreds)')
	#plt.title("Learning rate =" + str(learning_rates))
	legend = plt.legend(loc='upper center', shadow=True)
	frame = legend.get_frame()
	frame.set_facecolor('0.90')
	plt.show()


