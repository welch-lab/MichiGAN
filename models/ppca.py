#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg
import logging
import tensorflow as tf
from tensorflow import distributions as ds

from sklearn.decomposition import PCA
from .util import tf_log, logsumexp, total_correlation, shuffle_adata

log = logging.getLogger(__file__)


class ProbabilisticPCA:
	"""
	Probabilistic PCA object from the fitted PCA object, 
	self.z_mean is for the sampled representation using diagonized covariance matrix, 
	self.z_multivar_mean is with the fully-specified covariance matrix 
	"""

	def __init__(self, pca_fit):
		super().__init__()
		self.pca_fit = pca_fit
		self.z_dim = self.pca_fit.n_components_
		self.Wmatrix = self.pca_fit.components_.T
		self.pca_posterior()
		self.sess = tf.Session()
		self.mu = tf.placeholder(tf.float32, shape = (None, self.z_dim))
		self.z_multivar_mean = tf.placeholder(tf.float32, shape = (None, self.z_dim))
		self.pca_tensorflow()

	def pca_posterior(self):
		"""Compute the posterior mean and variance of principal components. 

		Returns
		-------
		Mmatrix : array, shape=(n_features, n_features)
			M matrix 
		MintWT: the matrix to compute the posterior mean 
		post_var: posterior variance of the pca
		"""
		n_features = self.pca_fit.components_.shape[1]

		# handle corner cases first
		if self.pca_fit.n_components_ == 0:
			return np.eye(n_features) / self.pca_fit.noise_variance_
		if self.pca_fit.n_components_ == n_features:
			return linalg.inv(self.pca_fit.get_covariance())

		# Get precision using matrix inversion lemma
		components_ = self.pca_fit.components_
		exp_var = self.pca_fit.explained_variance_
		if self.pca_fit.whiten:
			components_ = components_ * np.sqrt(exp_var[:, np.newaxis])
		exp_var_diff = np.maximum(exp_var - self.pca_fit.noise_variance_, 0.)
		
		precision = np.dot(components_, components_.T) / self.pca_fit.noise_variance_
		precision.flat[::len(precision) + 1] += 1. / exp_var_diff
		
		self.Mmatrix = precision.copy()
		self.MinvWT = np.dot(linalg.inv(precision), components_)
		self.post_var = precision/(self.pca_fit.noise_variance_**2)

	def encode_mean(self, x_data):
		"""
		encode data to the latent means
		"""
		Xr = x_data - self.pca_fit.mean_
		return np.dot(Xr, self.MinvWT.T)

	def log_prob_z_vector_post(self, z_vector):
		"""
		log probabilities of latent variables on given posterior normal distributions
		"""
		ll_con_dist = ds.Normal(self.mu, self.std)
		con_gen = ll_con_dist.log_prob(z_vector)

		return con_gen

	def pca_sample(self, mean_tensor, std_value, sample_size = 1):
		"""
		sample the posterior latent samples for representation
		"""
		ll_dist = ds.Normal(mean_tensor, std_value)
		ll_sample = ll_dist.sample(sample_size)
		ll_sample = tf.reshape(ll_sample, tf.shape(ll_sample)[1:])

		return ll_sample

	def pca_tensorflow(self):
		"""
		construct the PCA tensors
		"""
		
		self.std = tf.convert_to_tensor(np.sqrt(self.post_var.diagonal()))
		self.z_mean = self.pca_sample(self.mu, self.std)
		self.z_marginal_entropy, self.z_joint_entropy = self.qz_entropies()
		self.z_tc = total_correlation(self.z_marginal_entropy, self.z_joint_entropy)


	def qz_entropies(self):
		"""
		estimate the large sample entropies of the q(Z) and q(Zj)
		"""

		weights = - tf_log(tf.to_float(tf.shape(self.mu)[0]))

		function_to_map = lambda x: self.log_prob_z_vector_post(tf.reshape(x,[1, self.z_dim])) 
		logqz_i_m = tf.map_fn(function_to_map, self.z_mean, dtype = tf.float32)
		logqz_i_margin = logsumexp(logqz_i_m + weights, dim = 1, keepdims = False)
		logqz_value = tf.reduce_sum(logqz_i_m, axis = 2, keepdims = False)
		logqz_v_joint = logsumexp(logqz_value + weights, dim = 1, keepdims = False)
		logqz_sum = logqz_v_joint
		logqz_i_sum = logqz_i_margin

		marginal_entropies = (- tf.reduce_mean(logqz_i_sum, axis = 0))
		joint_entropies = (- tf.reduce_mean(logqz_sum)) 

		return marginal_entropies, joint_entropies

	def encode(self, x_data):
		"""
		encode data to the latent samples
		"""
		z_post_mean = self.encode_mean(x_data)
		z_post_var = self.post_var


		z_data = None
		for t in range(len(z_post_mean)):
			z_row = z_post_mean[t]
			if z_data is None:
				z_data = np.random.multivariate_normal(z_row, z_post_var).reshape((1, self.z_dim))
			else:
				z_sample = np.random.multivariate_normal(z_row, z_post_var).reshape((1, self.z_dim))
				z_data = np.concatenate((z_data, z_sample), axis = 0)

		return z_data

	def decode(self, z_data):
		"""
		decode latent samples to reconstructed data
		tensorflow makes the computation faster
		"""
		# x_dim = self.Wmatrix.shape[0]

		# mu_x = np.dot(z_data, self.Wmatrix.T) + self.pca_fit.mean_
		# std_x = self.pca_fit.noise_variance_**2
		# sigma_x = np.zeros((x_dim, x_dim))
		# np.fill_diagonal(sigma_x, std_x)
		
		# x_rec_data = None
		# for t in range(len(mu_x)):
		# 	mu_row = mu_x[t]
		# 	if x_rec_data is None:
		# 		x_sample = np.random.multivariate_normal(mu_row, sigma_x)
		# 		x_rec_data = x_sample.reshape((1, x_dim))
		# 	else:
		# 		x_sample = np.random.multivariate_normal(mu_row, sigma_x).reshape((1, x_dim))
		# 		x_rec_data = np.concatenate((x_rec_data, x_sample), axis = 0)

		# return x_rec_data
		x_dim = self.Wmatrix.shape[0]
		self.W = tf.convert_to_tensor(self.Wmatrix)
		self.mean_x = tf.convert_to_tensor(self.pca_fit.mean_)
	
		self.mu_x = tf.tensordot(self.z_multivar_mean, tf.transpose(self.W), axes = 1) + self.mean_x
		self.scale_x =  tf.convert_to_tensor(np.repeat(self.pca_fit.noise_variance_, x_dim))

		self.sigma_x = tf.reshape(tf.tile(self.scale_x, [tf.shape(self.mu_x)[0]]), 
			[tf.shape(self.mu_x)[0], tf.shape(self.scale_x)[0]])
		
		self.sigma_x = tf.cast(self.sigma_x, tf.float32)
		self.x_rec = self.pca_sample(self.mu_x, self.sigma_x)

		feed_dict = {self.z_multivar_mean: z_data}
		x_rec_data = self.sess.run(self.x_rec, feed_dict = feed_dict)

		return x_rec_data

	def reconstruct(self, x_data):
		"""
		reconstruct data from original data
		"""
		z_data = self.encode(x_data)
		x_rec_data = self.decode(z_data)

		return x_rec_data




		




