#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import time
import logging

import numpy as np
from scipy import sparse
import pandas as pd

import tensorflow as tf
from tensorflow import distributions as ds

from .util import *
from .Adam_prediction import Adam_Prediction_Optimizer
from .gan import BaseGAN

from metrics.GenerationMetrics import *
log = logging.getLogger(__file__)



class InfoWGAN_GP(BaseGAN):

	"""
	Information Maximizing Wasserstein GAN with gradient penalty (InfoWGAN-GP)
	"""
	def __init__(self, x_dimension, z_dimension = 10, noise_dimension = 118, **kwargs):

		super().__init__(x_dimension, z_dimension, **kwargs)

		self.n_dim = noise_dimension
		self.noise = tf.placeholder(tf.float32, shape = [None, self.n_dim], name = "latent_noise")
		self.lamb_mi_fake = kwargs.get("lamb_mi_fake", 1.0)

		with tf.device(self.device):

			self.x = tf.placeholder(tf.float32, shape = [None, self.x_dim], name = "data")
			self.z = tf.placeholder(tf.float32, shape = [None, self.z_dim], name = "latent")
			self.noise = tf.placeholder(tf.float32, shape = [None, self.n_dim], name = "latent_noise")

			self.create_network()
			self.loss_function()

		config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
		config.gpu_options.per_process_gpu_memory_fraction = 0.6
		self.sess = tf.Session(config = config)
		self.saver = tf.train.Saver(max_to_keep = 1)
		self.init = tf.global_variables_initializer().run(session = self.sess)
		

	def generatorDropOut(self):
		"""
		generator with dropout layers of InfoWGAN-GP
		"""
		with tf.variable_scope('generatorDropOut', reuse = tf.AUTO_REUSE):
			
			znoise = tf.concat([self.z, self.noise], axis = 1)

			ge_dense1 = tf.layers.dense(inputs = znoise, units = self.inflate_to_size1, activation = None, 
										kernel_initializer = self.init_w)
			ge_dense1 = tf.layers.batch_normalization(ge_dense1, training = self.is_training)
			ge_dense1 = tf.nn.leaky_relu(ge_dense1)
			ge_dense1 = tf.layers.dropout(ge_dense1, self.dropout_rate, training=self.is_training)

			ge_dense2 = tf.layers.dense(inputs = ge_dense1, units = self.inflate_to_size2, activation=None,
										kernel_initializer = self.init_w)
			ge_dense2 = tf.layers.batch_normalization(ge_dense2, training = self.is_training)
			ge_dense2 = tf.nn.leaky_relu(ge_dense2)
			ge_dense2 = tf.layers.dropout(ge_dense2, self.dropout_rate, training=self.is_training)

			ge_dense3 = tf.layers.dense(inputs = ge_dense2, units = self.inflate_to_size3, activation=None,
										kernel_initializer = self.init_w)
			ge_dense3 = tf.layers.batch_normalization(ge_dense3, training = self.is_training)
			ge_dense3 = tf.nn.relu(ge_dense3)
			ge_dense3 = tf.layers.dropout(ge_dense3, self.dropout_rate, training=self.is_training)

			ge_output = tf.layers.dense(inputs = ge_dense3, units= self.x_dim, activation=None)

			return ge_output   

	def generator(self):
		"""
		generator without dropout layers of InfoWGAN-GP
		"""
		with tf.variable_scope('generator', reuse = tf.AUTO_REUSE):

			znoise = tf.concat([self.z, self.noise], axis = 1)

			ge_dense1 = tf.layers.dense(inputs = znoise, units = self.inflate_to_size1, activation = None, 
										kernel_initializer = self.init_w)
			ge_dense1 = tf.layers.batch_normalization(ge_dense1, training = self.is_training)
			ge_dense1 = tf.nn.leaky_relu(ge_dense1)

			ge_dense2 = tf.layers.dense(inputs = ge_dense1, units = self.inflate_to_size2, activation=None,
										kernel_initializer = self.init_w)
			ge_dense2 = tf.layers.batch_normalization(ge_dense2, training = self.is_training)
			ge_dense2 = tf.nn.leaky_relu(ge_dense2)

			ge_dense3 = tf.layers.dense(inputs = ge_dense2, units = self.inflate_to_size3, activation=None,
										kernel_initializer = self.init_w)
			ge_dense3 = tf.layers.batch_normalization(ge_dense3, training = self.is_training)
			ge_dense3 = tf.nn.relu(ge_dense3)

			ge_output = tf.layers.dense(inputs = ge_dense3, units= self.x_dim, activation=None)

			return ge_output   
	

	def discriminatorInfo(self, x_input):
		"""
		discriminator of InfoWGAN-GP
		"""
		with tf.variable_scope('discriminatorInfo', reuse = tf.AUTO_REUSE):
			
			disc_dense1 = tf.layers.dense(inputs= x_input, units= self.disc_internal_size1, activation = None,
										  kernel_regularizer = self.regu_w, kernel_initializer = self.init_w)
			disc_dense1 = tf.layers.batch_normalization(disc_dense1, training = self.is_training)
			disc_dense1 = tf.nn.leaky_relu(disc_dense1)

			disc_dense2 = tf.layers.dense(inputs = disc_dense1, units= self.disc_internal_size2, activation=None,
										  kernel_regularizer = self.regu_w, kernel_initializer = self.init_w)
			disc_dense2 = tf.layers.batch_normalization(disc_dense2, training = self.is_training)
			disc_dense2 = tf.nn.leaky_relu(disc_dense2)

			disc_dense3 = tf.layers.dense(inputs=disc_dense2, units= self.disc_internal_size3, activation=None,
										  kernel_regularizer = self.regu_w, kernel_initializer = self.init_w)
			disc_dense3 = tf.layers.batch_normalization(disc_dense3, training = self.is_training)
			disc_dense3 = tf.nn.relu(disc_dense3)

			disc_output = tf.layers.dense(inputs=disc_dense3, units= 1, activation=None,
										  kernel_initializer = self.init_w)

			# Q network part
			q_dense1 = tf.layers.dense(inputs = disc_dense3, units = self.disc_internal_size3, activation = None,
									   kernel_initializer = self.init_w)
			q_dense1 = tf.layers.batch_normalization(q_dense1, training = self.is_training)
			q_dense1 = tf.nn.leaky_relu(q_dense1)

			q_output = tf.layers.dense(inputs = q_dense1, 
				units= (self.z_dim if self.fix_std else self.z_dim * 2), 
				activation = None)


			return disc_output, q_output  

	def c_mutual_info(self, c_vector, z_sample):
		"""
		function to compute the mutual information lower bound
		"""
		c_sample = z_sample[:, :self.z_dim]

		if self.fix_std:
			mean_vec = c_vector
			std_vec = tf.ones_like(mean_vec)
		else:
			mean_vec = c_vector[:, self.z_dim]
			std_vec = c_vector[:, self.z_dim:(self.z_dim * 2)]
			std_vec = tf.nn.softplus(std_vec)
		dist_c_vector = ds.Normal(mean_vec, std_vec)

		ll_logp = dist_c_vector.log_prob(c_sample)
		ll_logp_sum = tf.reduce_sum(ll_logp, [1])
		
		mi_bound = tf.reduce_mean(ll_logp_sum)

		return mi_bound
		
	def create_network(self):
		"""
		construct the InfoWGAN-GP networks
		"""
			
		if self.if_dropout:
			self.x_gen_data = self.generatorDropOut()
		else:
			self.x_gen_data = self.generator()

		self.Dx_real, self.c_mutual_real = self.discriminatorInfo(self.x)
		self.Dx_fake, self.c_mutual_fake = self.discriminatorInfo(self.x_gen_data)
		
		self.q_vector = self.c_mutual_info(self.c_mutual_fake, self.z)
		self.q_fake_mutual = tf.reduce_mean(self.q_vector)

	def compute_gp(self, x, x_gen_data, discriminator):
		"""
		gradient penalty of discriminator
		"""
		epsilon_x = tf.random_uniform([], 0.0, 1.0)
		x_hat = x * epsilon_x + (1 - epsilon_x) * x_gen_data
		d_hat, _ = discriminator(x_hat)

		gradients = tf.gradients(d_hat, x_hat)[0]
		slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
		
		gradient_penalty =  tf.reduce_mean((slopes - 1.0) ** 2)
		
		return gradient_penalty

	def loss_function(self):
		"""
		loss function of InfoWGAN-GP
		"""

		D_raw_loss = tf.reduce_mean(self.Dx_real) - tf.reduce_mean(self.Dx_fake)
		self.G_loss = tf.reduce_mean(self.Dx_fake) - (self.q_fake_mutual * self.lamb_mi_fake)
		
		self.gradient_penalty = self.compute_gp(self.x, self.x_gen_data, self.discriminatorInfo)
		self.D_loss = D_raw_loss + self.lamb_gp * self.gradient_penalty - (self.q_fake_mutual * self.lamb_mi_fake)
		
		tf_vars_all = tf.trainable_variables()
		dvars  = [var for var in tf_vars_all if var.name.startswith("discriminatorInfo")]

		if self.if_dropout:
			gvars  = [var for var in tf_vars_all if var.name.startswith("generatorDropOut")]
		else:
			gvars  = [var for var in tf_vars_all if var.name.startswith("generator")]

		self.parameter_count = tf.reduce_sum(
			[tf.reduce_prod(tf.shape(v)) for v in dvars + gvars])

		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.g_solver = Adam_Prediction_Optimizer(learning_rate = self.learning_rate, 
				beta1=0.9, beta2=0.999, prediction=True).minimize(self.G_loss, var_list = gvars)
			self.d_solver = Adam_Prediction_Optimizer(learning_rate = self.learning_rate, 
				beta1=0.9, beta2=0.999, prediction=False).minimize(self.D_loss, var_list = dvars)
	
	@property
	def model_parameter(self):
		"""
		report the number of training parameters
		"""
		self.total_param = self.sess.run(self.parameter_count)
		return "There are {} parameters in InfoWGAN-GP.".format(self.total_param)


	def generate_cells(self, z_data):
		"""
		generate data from latent samples
		"""
		noise_data = self.sample_z(len(z_data), self.n_dim)
		gen_data = self.sess.run(self.x_gen_data, feed_dict = {self.z: z_data, 
			self.noise: noise_data, self.is_training: False})
		return gen_data


	def restore_model(self, model_path):
		"""
		restore model from model_path
		"""
		self.saver.restore(self.sess, model_path)


	def save_model(self, model_save_path, epoch):
		"""
		save the trained model to the model_save_path
		"""
		os.makedirs(model_save_path, exist_ok = True)
		model_save_name = os.path.join(model_save_path, "model")
		save_path = self.saver.save(self.sess, model_save_name, global_step = epoch)

		np.save(os.path.join(model_save_path, "training_time.npy"), self.training_time)
		np.save(os.path.join(model_save_path, "train_loss_D.npy"), self.train_loss_D)
		np.save(os.path.join(model_save_path, "train_loss_G.npy"), self.train_loss_G)
		np.save(os.path.join(model_save_path, "valid_loss_D.npy"), self.valid_loss_D)
		np.save(os.path.join(model_save_path, "valid_loss_G.npy"), self.valid_loss_G)

	def train(self, train_data, use_validation = False, valid_data = None, use_test_during_train = False, test_data = None,
			  test_every_n_epochs = 100, test_size = 3000, inception_score_data = None, n_epochs = 25, batch_size = 32, 
			  early_stop_limit = 20, threshold = 0.0025, shuffle = True, save = False, model_save_path = None, output_save_path = None, 
			  verbose = False):
		"""
		train InfoWGAN-GP with train_data (AnnData) and optional valid_data (numpy array) for n_epochs. 
		"""
		log.info("--- Training ---")
		if use_validation and valid_data is None:
			raise Exception('valid_data is None but use_validation is True.')
		
		patience = early_stop_limit
		min_delta = threshold
		patience_cnt = 0

		n_train = train_data.shape[0]
		n_valid = None
		if use_validation:
			n_valid = valid_data.shape[0]

		# generation performance at the PC space
		if use_test_during_train:
			pca_data_50 = PCA(n_components = 50, random_state = 42)
			genmetric = MetricVisualize()
			RFE = RandomForestError()
			genmetrics_pd = pd.DataFrame({'epoch':[], 'is_real_mu': [], 'is_real_std': [], 
										  'is_fake_mu':[], 'is_fake_std':[], 'rf_error':[]})

			if sparse.issparse(train_data.X):
				pca_data_fit = pca_data_50.fit(train_data.X.A)
			else:
				pca_data_fit = pca_data_50.fit(train_data.X)

		train_data_copy = train_data.copy()
		for epoch in range(1, n_epochs + 1):
			begin = time.time()

			if shuffle: 
				train_data = shuffle_adata(train_data)
			train_loss_D, train_loss_G = 0.0, 0.0
			valid_loss_D, valid_loss_G = 0.0, 0.0

			for _ in range(1,  n_train // batch_size + 1):

				# D step
				for _ in range(self.Diters):
					x_mb = self.sample_data(train_data, batch_size)
					z_mb = self.sample_z(batch_size, self.z_dim)
					n_mb = self.sample_z(batch_size, self.n_dim)
					self.sess.run(self.d_solver, feed_dict = {self.x: x_mb, self.z: z_mb, self.noise: n_mb, self.is_training: self.if_BNTrainingMode})
				
				# G step
				x_mb = self.sample_data(train_data, batch_size)
				z_mb = self.sample_z(batch_size, self.z_dim)
				n_mb = self.sample_z(batch_size, self.n_dim)
				_, current_loss_D, current_loss_G = self.sess.run([self.g_solver, self.D_loss, self.G_loss],
					feed_dict = {self.x: x_mb, self.z: z_mb, self.noise: n_mb, self.is_training: self.if_BNTrainingMode})

				train_loss_D += (current_loss_D * batch_size)
				train_loss_G += (current_loss_G * batch_size)
			
			train_loss_D /= n_train
			train_loss_G /= n_train 

			if use_validation:
				for _ in range(1, n_valid // batch_size + 1):
					x_mb = self.sample_data(valid_data, batch_size)
					z_mb = self.sample_z(batch_size, self.z_dim)
					n_mb = self.sample_z(batch_size, self.n_dim)
					current_loss_valid_D, current_loss_valid_G = self.sess.run([self.D_loss, self.G_loss], 
						feed_dict = {self.x: x_mb, self.z: z_mb, self.noise: n_mb, self.is_training: False})

					valid_loss_D += current_loss_valid_D
					valid_loss_G += current_loss_valid_G
				
				valid_loss_D /= n_valid
				valid_loss_G /= n_valid
			
			self.train_loss_D.append(train_loss_D)
			self.train_loss_G.append(train_loss_G)
			self.valid_loss_D.append(valid_loss_D)
			self.valid_loss_G.append(valid_loss_G)
			self.training_time += (time.time() - begin)

			# testing for generation metrics
			if (epoch - 1) % test_every_n_epochs == 0 and use_test_during_train:
				
				if test_data is None:
					reset_test_data = True
					sampled_indices = sample(range(n_train), test_size)
					
					if sparse.issparse(train_data_copy.X):
						test_data = train_data_copy[sampled_indices, :].X.A
					else:
						test_data = train_data_copy[sampled_indices, :].X

					z_data = self.sample_z(test_size, self.z_dim)
					gen_data = self.generate_cells(z_data)

					if inception_score_data is not None:
						inception_score_subdata = inception_score_data[sampled_indices]
						mean_is_real, std_is_real = genmetric.InceptionScore(test_data, inception_score_subdata, test_data)
						mean_is_fake, std_is_fake = genmetric.InceptionScore(test_data, inception_score_subdata, gen_data)
					else:
						mean_is_real = std_is_real = mean_is_fake = std_is_fake = 0.0

				else:
					assert test_data.shape[0] == test_size
					reset_test_data = False

					z_data = self.sample_z(test_size, self.z_dim)
					gen_data = self.generate_cells(z_data)

					if inception_score_data is not None:
						inception_score_subdata = inception_score_data
						mean_is_real, std_is_real = genmetric.InceptionScore(test_data, inception_score_subdata, test_data)
						mean_is_fake, std_is_fake = genmetric.InceptionScore(test_data, inception_score_subdata, gen_data)
					else:
						mean_is_real = std_is_real = mean_is_fake = std_is_fake = 0.0


					

				errors_d = list(RFE.fit(test_data, gen_data, pca_data_fit, if_dataPC = True, output_AUC = False)['avg'])[0]
				genmetrics_pd = pd.concat([genmetrics_pd, pd.DataFrame([[epoch, mean_is_real, std_is_real, mean_is_fake, std_is_fake, 
						errors_d]], columns = ['epoch', 'is_real_mu', 'is_real_std', 'is_fake_mu', 'is_fake_std', 'rf_error'])])
				if save:
					genmetrics_pd.to_csv(os.path.join(model_save_path, "GenerationMetrics.csv"))

				if reset_test_data:
					test_data = None

			if verbose: 
				print(f"Epoch {epoch}: D Train Loss: {train_loss_D} G Train Loss: {train_loss_G}  D Valid Loss: {valid_loss_D} G Valid Loss: {valid_loss_G}")

			# early stopping
			if use_validation and epoch > 1:
				if abs(self.valid_loss_D[epoch - 2] - self.valid_loss_D[epoch - 1]) > min_delta or abs(self.valid_loss_G[epoch - 2] - self.valid_loss_G[epoch - 1]) > min_delta:
					patience_cnt = 0
				else:
					patience_cnt += 1

				if patience_cnt > patience:
					if save:
						self.save_model(model_save_path, epoch)
						log.info(f"Model saved in file: {model_save_path}. Training stopped earlier at epoch: {epoch}.")
						if verbose:
							print(f"Model saved in file: {model_save_path}. Training stopped earlier at epoch: {epoch}.")
						if use_test_during_train:
							genmetrics_pd.to_csv(os.path.join(model_save_path, "GenerationMetrics.csv"))
					break

		if save:
			self.save_model(model_save_path, epoch)
			log.info(f"Model saved in file: {model_save_path}. Training finished.")
			if verbose:
				print(f"Model saved in file: {model_save_path}. Training finished.")

			if use_test_during_train:
				genmetrics_pd.to_csv(os.path.join(model_save_path, "GenerationMetrics.csv"))


	def train_np(self, train_data, use_validation = False, valid_data = None, use_test_during_train = False, test_data = None,
				 test_every_n_epochs = 100, test_size = 3000, inception_score_data = None, n_epochs = 25, batch_size = 32, 
				 early_stop_limit = 20, threshold = 0.0025, shuffle = True, save = False, model_save_path = None, output_save_path = None, 
				 verbose = False):
		"""
		train InfoWGAN-GP with train_data (numpy array) and optional valid_data (numpy array) for n_epochs.
		"""
		log.info("--- Training ---")
		if use_validation and valid_data is None:
			raise Exception('valid_data is None but use_validation is True.')
		
		patience = early_stop_limit
		min_delta = threshold
		patience_cnt = 0

		n_train = train_data.shape[0]
		n_valid = None
		if use_validation:
			n_valid = valid_data.shape[0]

		# generation performance at the PC space
		if use_test_during_train:
			pca_data_50 = PCA(n_components = 50, random_state = 42)
			genmetric = MetricVisualize()
			RFE = RandomForestError()
			genmetrics_pd = pd.DataFrame({'epoch':[], 'is_real_mu': [], 'is_real_std': [], 
										  'is_fake_mu':[], 'is_fake_std':[], 'rf_error':[]})

			pca_data_fit = pca_data_50.fit(train_data)

		if shuffle:
			index_shuffle = list(range(n_train))

		for epoch in range(1, n_epochs + 1):

			begin = time.time()

			if shuffle:
				np.random.shuffle(index_shuffle)
				train_data = train_data[index_shuffle]
				if inception_score_data is not None:
					inception_score_data = inception_score_data[index_shuffle]

			train_loss_D, train_loss_G = 0.0, 0.0
			valid_loss_D, valid_loss_G = 0.0, 0.0

			for _ in range(1,  n_train // batch_size + 1):

				# D step
				for _ in range(self.Diters):
					x_mb = self.sample_data_np(train_data, batch_size)
					z_mb = self.sample_z(batch_size, self.z_dim)
					n_mb = self.sample_z(batch_size, self.n_dim)
					self.sess.run(self.d_solver, feed_dict = {self.x: x_mb, self.z: z_mb, self.noise: n_mb, 
															  self.is_training: self.if_BNTrainingMode})
				
				# G step
				x_mb = self.sample_data_np(train_data, batch_size)
				z_mb = self.sample_z(batch_size, self.z_dim)
				n_mb = self.sample_z(batch_size, self.n_dim)
				_, current_loss_D, current_loss_G = self.sess.run([self.g_solver, self.D_loss, self.G_loss],
					feed_dict = {self.x: x_mb, self.z: z_mb, self.noise: n_mb, self.is_training: self.if_BNTrainingMode})

				train_loss_D += (current_loss_D * batch_size)
				train_loss_G += (current_loss_G * batch_size)
			
			train_loss_D /= n_train
			train_loss_G /= n_train 

			if use_validation:
				for _ in range(1, n_valid // batch_size + 1):
					x_mb = self.sample_data_np(valid_data, batch_size)
					z_mb = self.sample_z(batch_size, self.z_dim)
					n_mb = self.sample_z(batch_size, self.n_dim)

					current_loss_valid_D, current_loss_valid_G = self.sess.run([self.D_loss, self.G_loss], 
						feed_dict = {self.x: x_mb, self.z: z_mb, self.noise: n_mb, self.is_training: False})

					valid_loss_D += current_loss_valid_D
					valid_loss_G += current_loss_valid_G
				
				valid_loss_D /= n_valid
				valid_loss_G /= n_valid
			
			self.train_loss_D.append(train_loss_D)
			self.train_loss_G.append(train_loss_G)
			self.valid_loss_D.append(valid_loss_D)
			self.valid_loss_G.append(valid_loss_G)
			self.training_time += (time.time() - begin)

			# testing for generation metrics
			if (epoch - 1) % test_every_n_epochs == 0 and use_test_during_train:
				
				if test_data is None:
					reset_test_data = True
					sampled_indices = sample(range(n_train), test_size)
					
					test_data = train_data[sampled_indices, :]

					z_data = self.sample_z(test_size, self.z_dim)
					gen_data = self.generate_cells(z_data)

					if inception_score_data is not None:
						inception_score_subdata = inception_score_data[sampled_indices]
						mean_is_real, std_is_real = genmetric.InceptionScore(test_data, inception_score_subdata, test_data)
						mean_is_fake, std_is_fake = genmetric.InceptionScore(test_data, inception_score_subdata, gen_data)
					else:
						mean_is_real = std_is_real = mean_is_fake = std_is_fake = 0.0

				else:
					assert test_data.shape[0] == test_size
					reset_test_data = False

					z_data = self.sample_z(test_size, self.z_dim)
					gen_data = self.generate_cells(z_data)

					if inception_score_data is not None:
						inception_score_subdata = inception_score_data
						mean_is_real, std_is_real = genmetric.InceptionScore(test_data, inception_score_subdata, test_data)
						mean_is_fake, std_is_fake = genmetric.InceptionScore(test_data, inception_score_subdata, gen_data)
					else:
						mean_is_real = std_is_real = mean_is_fake = std_is_fake = 0.0

				errors_d = list(RFE.fit(test_data, gen_data, pca_data_fit, if_dataPC = True, output_AUC = False)['avg'])[0]
				genmetrics_pd = pd.concat([genmetrics_pd, pd.DataFrame([[epoch, mean_is_real, std_is_real, mean_is_fake, std_is_fake, 
						errors_d]], columns = ['epoch', 'is_real_mu', 'is_real_std', 'is_fake_mu', 'is_fake_std', 'rf_error'])])
				if save:
					genmetrics_pd.to_csv(os.path.join(output_save_path, "GenerationMetrics.csv"))

				if reset_test_data:
					test_data = None

			if verbose: 
				print(f"Epoch {epoch}: D Train Loss: {train_loss_D} G Train Loss: {train_loss_G}  D Valid Loss: {valid_loss_D} G Valid Loss: {valid_loss_G}")

			# early stopping
			if use_validation and epoch > 1:
				if abs(self.valid_loss_D[epoch - 2] - self.valid_loss_D[epoch - 1]) > min_delta or abs(self.valid_loss_G[epoch - 2] - self.valid_loss_G[epoch - 1]) > min_delta:
					patience_cnt = 0
				else:
					patience_cnt += 1

				if patience_cnt > patience:
					if save:
						self.save_model(model_save_path, epoch)
						log.info(f"Model saved in file: {model_save_path}. Training stopped earlier at epoch: {epoch}.")
						if verbose:
							print(f"Model saved in file: {model_save_path}. Training stopped earlier at epoch: {epoch}.")
						if use_test_during_train:
							genmetrics_pd.to_csv(os.path.join(model_save_path, "GenerationMetrics.csv"))
					break

		if save:
			self.save_model(model_save_path, epoch)
			log.info(f"Model saved in file: {model_save_path}. Training finished.")
			if verbose:
				print(f"Model saved in file: {model_save_path}. Training finished.")

			if use_test_during_train:
				genmetrics_pd.to_csv(os.path.join(model_save_path, "GenerationMetrics.csv"))
