from gensim.models import *
import numpy as np
import re
import sys
import time
import collections
import matplotlib.pyplot as plt
from scipy.linalg import orth
from embeddings import *


def initialize(dim_D, dim_W):
	dictionary = np.zeros((dim_D, dim_D))
	for i in range(dim_D):
		dictionary[i][i] = 1
	W = np.zeros((dim_W, dim_W))
	#W = np.random.randn(dim_W, dim_W)
	return dictionary, W

def emb_normalize():
	srcfile = open("model_EN/v7_EN.vector", encoding='utf-8', errors='surrogateescape')
	trgfile = open("model_ES/v7_ES.vector", encoding='utf-8', errors='surrogateescape')
	src_words, x = read(srcfile, dtype='float')
	print('英文模型加载完毕！')
	trg_words, z = read(trgfile, dtype='float')
	print('西班牙文模型加载完毕！')
	src_word2ind = {word: i for i, word in enumerate(src_words)}
	trg_word2ind = {word: i for i, word in enumerate(trg_words)}
	x = length_normalize(x)
	x = mean_center(x)
	z = length_normalize(z)
	z = mean_center(z)
	srcfile = open("model_EN/v7_EN_nor.vec", mode='w', encoding='utf-8', errors='surrogateescape')
	trgfile = open("model_ES/v7_ES_nor.vec", mode='w', encoding='utf-8', errors='surrogateescape')
	write(src_words, x, srcfile)
	write(trg_words, z, trgfile)
	srcfile.close()
	trgfile.close()



def dropout(m, p):
	if p <= 0.0:
		return m
	else:
		mask = np.random.rand(*m.shape) >= p
		return m*mask

def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
	n = m.shape[0]
	ans = np.zeros(n, dtype=m.dtype)
	if k <= 0:
		return ans
	if not inplace:
		m = np.array(m)
	ind0 = np.arange(n)
	ind1 = np.empty(n, dtype=int)
	minimum = m.min()
	for i in range(k):
		m.argmax(axis=1, out=ind1)
		ans += m[ind0, ind1]
		m[ind0, ind1] = minimum
	return ans / k

def main(style):
	# emb_normalize()matrix
	#model_EN = KeyedVectors.load_word2vec_format('model_EN/v7_EN_normal.vec', binary=False)
	#print('英文模型加载完毕！')
	#model_IT = KeyedVectors.load_word2vec_format('model_ES/v7_ES_normal.vec', binary=False)
	#print('西班牙文模型加载完毕！')
	#vocab = np.load('vocab/vocabEN-ES.npy')
	srcfile = open('model_EN/v7_EN_nor.vec', encoding='utf-8', errors='surrogateescape')
	trgfile = open('model_ES/v7_ES_nor.vec', encoding='utf-8', errors='surrogateescape')
	batch_size = 1000
	validation = None
	log = 'log/log.txt'
	src_words, x = read(srcfile, dtype='float')
	trg_words, z = read(trgfile, dtype='float')
	datatype = x.dtype
	src_word2ind = {word: i for i, word in enumerate(src_words)}
	trg_word2ind = {word: i for i, word in enumerate(trg_words)}


	# Build the seed dictionary
	src_indices = []
	trg_indices = []
	if style == 'unsupervised':
		sim_size = min(x.shape[0], z.shape[0])
		u, s, vt = np.linalg.svd(x[:sim_size], full_matrices=False)
		x_sim = (u * s).dot(u.T)
		print('源语言相似矩阵构建完毕')
		u, s, vt = np.linalg.svd(z[:sim_size], full_matrices=False)
		z_sim = (u * s).dot(u.T)
		print('目标语言相似矩阵构建完毕')
		del u, s, vt
		x_sim.sort(axis=1)
		z_sim.sort(axis=1)
		x_pie = length_normalize(x_sim)
		z_pie = length_normalize(z_sim)

	elif style == 'semi-supervised':
		vocab = np.load('vocab/vocabEN-ES.npy')
		for i in range(25):
			src_ind = src_word2ind[vocab[i][0]]
			trg_ind = trg_word2ind[vocab[i][1]]
			src_indices.append(src_ind)
			trg_indices.append(trg_ind)
	else:
		print('ERROR!')

	# Read validation dictionary
	if validation is not None:
		f = open(validation, encoding='utf-8', errors='surrogateescape')
		validation = collections.defaultdict(set)
		oov = set()
		vocab = set()
		for line in f:
			src, trg = line.split()
			try:
				src_ind = src_word2ind[src]
				trg_ind = trg_word2ind[trg]
				validation[src_ind].add(trg_ind)
				vocab.add(src)
			except KeyError:
				oov.add(src)
		oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
		validation_coverage = len(validation) / (len(validation) + len(oov))

	# Create log file
	if log:
		logfile = open(log, mode='w', encoding='utf-8', errors='surrogateescape')
	# Allocate memory
	vocabulary_cutoff = 20000
	xw = np.empty_like(x)
	zw = np.empty_like(z)
	src_size = x.shape[0] if vocabulary_cutoff <= 0 else min(x.shape[0], vocabulary_cutoff)
	trg_size = z.shape[0] if vocabulary_cutoff <= 0 else min(z.shape[0], vocabulary_cutoff)
	simfwd = np.empty((batch_size, trg_size), dtype=datatype)
	simbwd = np.empty((batch_size, src_size), dtype=datatype)
	if validation is not None:
		simval = np.empty((len(validation.keys()), z.shape[0]), dtype=float)

	best_sim_forward = np.full(src_size, -100, dtype=datatype)
	src_indices_forward = np.arange(src_size)
	trg_indices_forward = np.zeros(src_size, dtype=int)
	best_sim_backward = np.full(trg_size, -100, dtype=datatype)
	src_indices_backward = np.zeros(trg_size, dtype=int)
	trg_indices_backward = np.arange(trg_size)
	knn_sim_fwd = np.zeros(src_size, dtype=datatype)
	knn_sim_bwd = np.zeros(trg_size, dtype=datatype)

	# Training loop
	orthogonal = True
	whiten = True
	unconstrained = True
	src_reweight = 0.5
	trg_reweight = 0.5
	dim_reduction = 0
	best_objective = objective = -100.
	it = 1
	last_improvement = 0
	keep_prob = 0.1  # 0.1
	direction = 'union'
	csls_neighborhood = 10
	threshold = 0.0001
	verbose = True
	t = time.time()
	end = False
	while True:

		# Increase the keep probability if we have not improve in args.stochastic_interval iterations
		if it - last_improvement > 50:  # 50, args.stochastic_interval
			if keep_prob >= 1.0:
				end = True
			keep_prob = min(1.0, 2 * keep_prob)
			last_improvement = it

		# Update the embedding mapping
		if orthogonal or not end:  # orthogonal mapping
			u, s, vt = np.linalg.svd(z[trg_indices].T.dot(x[src_indices]))
			w = vt.T.dot(u.T)
			x.dot(w, out=xw)
			zw[:] = z
		elif unconstrained:  # unconstrained mapping
			x_pseudoinv = np.linalg.inv(x[src_indices].T.dot(x[src_indices])).dot(x[src_indices].T)
			w = x_pseudoinv.dot(z[trg_indices])
			x.dot(w, out=xw)
			zw[:] = z
		else:  # advanced mapping

			# TODO xw.dot(wx2, out=xw) and alike not working
			xw[:] = x
			zw[:] = z

			# STEP 1: Whitening
			def whitening_transformation(m):
				u, s, vt = np.linalg.svd(m, full_matrices=False)
				return vt.T.dot(np.diag(1 / s)).dot(vt)

			if whiten:
				wx1 = whitening_transformation(xw[src_indices])
				wz1 = whitening_transformation(zw[trg_indices])
				xw = xw.dot(wx1)
				zw = zw.dot(wz1)

			# STEP 2: Orthogonal mapping
			wx2, s, wz2_t = np.linalg.svd(xw[src_indices].T.dot(zw[trg_indices]))
			wz2 = wz2_t.T
			xw = xw.dot(wx2)
			zw = zw.dot(wz2)

			# STEP 3: Re-weighting
			xw *= s ** src_reweight
			zw *= s ** trg_reweight

			# STEP 4: De-whitening
			xw = xw.dot(wx2.T.dot(np.linalg.inv(wx1)).dot(wx2))
			zw = zw.dot(wz2.T.dot(np.linalg.inv(wz1)).dot(wz2))

			# STEP 5: Dimensionality reduction
			if dim_reduction > 0:
				xw = xw[:, :dim_reduction]
				zw = zw[:, :dim_reduction]

		# Self-learning
		if end:
			break
		else:
			# Update the training dictionary
			if direction in ('forward', 'union'):
				if csls_neighborhood > 0:
					for i in range(0, trg_size, simbwd.shape[0]):
						j = min(i + simbwd.shape[0], trg_size)
						zw[i:j].dot(xw[:src_size].T, out=simbwd[:j - i])
						knn_sim_bwd[i:j] = topk_mean(simbwd[:j - i], k=csls_neighborhood, inplace=True)
				for i in range(0, src_size, simfwd.shape[0]):
					j = min(i + simfwd.shape[0], src_size)
					xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j - i])
					simfwd[:j - i].max(axis=1, out=best_sim_forward[i:j])
					simfwd[:j - i] -= knn_sim_bwd / 2  # Equivalent to the real CSLS scores for NN
					dropout(simfwd[:j - i], 1 - keep_prob).argmax(axis=1, out=trg_indices_forward[i:j])
			if direction in ('backward', 'union'):
				if csls_neighborhood > 0:
					for i in range(0, src_size, simfwd.shape[0]):
						j = min(i + simfwd.shape[0], src_size)
						xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j - i])
						knn_sim_fwd[i:j] = topk_mean(simfwd[:j - i], k=csls_neighborhood, inplace=True)
				for i in range(0, trg_size, simbwd.shape[0]):
					j = min(i + simbwd.shape[0], trg_size)
					zw[i:j].dot(xw[:src_size].T, out=simbwd[:j - i])
					simbwd[:j - i].max(axis=1, out=best_sim_backward[i:j])
					simbwd[:j - i] -= knn_sim_fwd / 2  # Equivalent to the real CSLS scores for NN
					dropout(simbwd[:j - i], 1 - keep_prob).argmax(axis=1, out=src_indices_backward[i:j])
			if direction == 'forward':
				src_indices = src_indices_forward
				trg_indices = trg_indices_forward
			elif direction == 'backward':
				src_indices = src_indices_backward
				trg_indices = trg_indices_backward
			elif direction == 'union':
				src_indices = np.concatenate((src_indices_forward, src_indices_backward))
				trg_indices = np.concatenate((trg_indices_forward, trg_indices_backward))

			# Objective function evaluation
			if direction == 'forward':
				objective = np.mean(best_sim_forward).tolist()
			elif direction == 'backward':
				objective = np.mean(best_sim_backward).tolist()
			elif direction == 'union':
				objective = (np.mean(best_sim_forward) + np.mean(best_sim_backward)).tolist() / 2
			if objective - best_objective >= threshold:
				last_improvement = it
				best_objective = objective

			# Accuracy and similarity evaluation in validation
			if validation is not None:
				src = list(validation.keys())
				xw[src].dot(zw.T, out=simval)
				nn = simval.argmax(axis=1)
				accuracy = np.mean([1 if nn[i] in validation[src[i]] else 0 for i in range(len(src))])
				similarity = np.mean(
					[max([simval[i, j].tolist() for j in validation[src[i]]]) for i in range(len(src))])

			# Logging
			duration = time.time() - t
			if verbose:
				print(file=sys.stderr)
				print('ITERATION {0} ({1:.2f}s)'.format(it, duration), file=sys.stderr)
				print('\t- Objective:        {0:9.4f}%'.format(100 * objective), file=sys.stderr)
				print('\t- Drop probability: {0:9.4f}%'.format(100 - 100 * keep_prob), file=sys.stderr)
				if validation is not None:
					print('\t- Val. similarity:  {0:9.4f}%'.format(100 * similarity), file=sys.stderr)
					print('\t- Val. accuracy:    {0:9.4f}%'.format(100 * accuracy), file=sys.stderr)
					print('\t- Val. coverage:    {0:9.4f}%'.format(100 * validation_coverage), file=sys.stderr)
				sys.stderr.flush()
			if log is not None:
				val = '{0:.6f}\t{1:.6f}\t{2:.6f}'.format(
					100 * similarity, 100 * accuracy,
					100 * validation_coverage) if validation is not None else ''
				print('{0}\t{1:.6f}\t{2}\t{3:.6f}'.format(it, 100 * objective, val, duration), file=logfile)
				logfile.flush()

		t = time.time()
		it += 1

	# Write mapped embeddings
	srcfile = open('src_mappedEMB', mode='w', encoding='utf-8', errors='surrogateescape')
	trgfile = open('trg_mappedEMB', mode='w', encoding='utf-8', errors='surrogateescape')
	write(src_words, xw, srcfile)
	write(trg_words, zw, trgfile)
	srcfile.close()
	trgfile.close()




if __name__ == '__main__':
	'''model_EN = KeyedVectors.load_word2vec_format('model_EN/EN.200K.cbow.txt', binary=False)
	print('英文模型加载完毕！')
	model_IT = KeyedVectors.load_word2vec_format('model_IT/IT.200K.cbow.txt', binary=False)
	print('意大利文模型加载完毕！')'''
	styles = ['unsupervised', 'semi-supervised']
	main(styles[1])


