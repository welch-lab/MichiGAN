#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import time
import logging

import numpy as np
from scipy import sparse
import pandas as pd

from .cgan import * 
from .ppca import * 
from .vae import *

log = logging.getLogger(__file__)


class MichiGAN:
	"""
	MichiGAN class:
		DisentangleMethod is pre-trained PCA, VAE or beta-TCVAE
		GenerationMethod is untrained GANs
	"""
	def __init__(self, DisentangleMethod, GenerationMethod):
		super().__init__()
		self.DisentangleMethod = DisentangleMethod
		self.GenerationMethod = GenerationMethod


	def train_mean(self, train_data, use_validation = False, valid_data = None, use_test_during_train = False, test_data = None,
				   test_every_n_epochs = 100, test_size = 3000, inception_score_data = None, n_epochs = 25, batch_size = 32, 
				   early_stop_limit = 20, threshold = 0.0025, shuffle = True, save = False, model_save_path = None, output_save_path = None, 
				   verbose = False):
		"""
		train GenerationMethod with latent representation conditions of posterior means 
		train_data (AnnData) and optional valid_data (numpy array) for n_epochs. 
		"""
		if sparse.issparse(train_data.X):
			train_dense = train_data.X.A
		else:
			train_dense = train_data.X

		train_cond = self.DisentangleMethod.encode_mean(train_dense)

		if use_validation:
			if sparse.issparse(valid_data.X):
				valid_dense = valid_data.X.A
			else:
				valid_dense = valid_data.X
			valid_cond = self.DisentangleMethod.encode_mean(valid_dense)
		else:
			valid_cond = None
		
		if test_data is not None:
			test_cond = self.DisentangleMethod.encode_mean(test_cond)
		else:
			test_cond = None

		self.GenerationMethod.train(train_data = train_data, train_cond = train_cond, use_validation = use_validation, 
									valid_data = valid_data, valid_cond = valid_cond, use_test_during_train = use_test_during_train, 
									test_data = test_data, test_cond = test_cond, test_every_n_epochs = test_every_n_epochs, 
									test_size = test_size, inception_score_data = inception_score_data, n_epochs = n_epochs, 
									batch_size = batch_size, early_stop_limit = early_stop_limit, threshold = threshold, 
									shuffle = shuffle, save = save, model_save_path = model_save_path, output_save_path = output_save_path, 
									verbose = verbose)

	def train_mean_np(self, train_data, use_validation = False, valid_data = None, use_test_during_train = False, test_data = None,
					  test_every_n_epochs = 100, test_size = 3000, inception_score_data = None, n_epochs = 25, batch_size = 32, 
					  early_stop_limit = 20, threshold = 0.0025, shuffle = True, save = False, model_save_path = None, output_save_path = None, 
					  verbose = False):
		"""
		train GenerationMethod with latent representation conditions of posterior means
		train_data (numpy array) and optional valid_data (numpy array) for n_epochs. 
		"""

		train_cond = self.DisentangleMethod.encode_mean(train_data)

		if use_validation:
			valid_cond = self.DisentangleMethod.encode_mean(valid_data)
		else:
			valid_cond = None
		
		if test_data is not None:
			test_cond = self.DisentangleMethod.encode_mean(test_data)
		else:
			test_cond = None

		self.GenerationMethod.train_np(train_data = train_data, train_cond = train_cond, use_validation = use_validation, 
									   valid_data = valid_data, valid_cond = valid_cond, use_test_during_train = use_test_during_train, 
									   test_data = test_data, test_cond = test_cond, test_every_n_epochs = test_every_n_epochs, 
									   test_size = test_size, inception_score_data = inception_score_data, n_epochs = n_epochs, 
									   batch_size = batch_size, early_stop_limit = early_stop_limit, threshold = threshold, 
									   shuffle = shuffle, save = save, model_save_path = model_save_path, output_save_path = output_save_path, 
									   verbose = verbose)

	def train_postvars_np(self, train_data, use_validation = False, valid_data = None, use_test_during_train = False, test_data = None,
						  test_every_n_epochs = 100, test_size = 3000, inception_score_data = None, n_epochs = 25, batch_size = 32, 
						  early_stop_limit = 20, threshold = 0.0025, shuffle = True, save = False, model_save_path = None, output_save_path = None, 
						  verbose = False):
		"""
		train GenerationMethod with latent representation conditions of posterior samples
		train_data (numpy array) and optional valid_data (numpy array) for n_epochs. 
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
				for _ in range(self.GenerationMethod.Diters):
					x_mb = self.GenerationMethod.sample_data_np(train_data, batch_size)
					z_mb = self.DisentangleMethod.encode(x_mb)
					n_mb = self.GenerationMethod.sample_z(batch_size, self.GenerationMethod.n_dim)
					self.GenerationMethod.sess.run(self.GenerationMethod.d_solver, 
												   feed_dict = {self.GenerationMethod.x: x_mb, 
																self.GenerationMethod.z: z_mb, 
																self.GenerationMethod.noise: n_mb, 
																self.GenerationMethod.is_training: self.GenerationMethod.if_BNTrainingMode})
				
				# G step
				x_mb = self.GenerationMethod.sample_data_np(train_data, batch_size)
				z_mb = self.DisentangleMethod.encode(x_mb)
				n_mb = self.GenerationMethod.sample_z(batch_size, self.GenerationMethod.n_dim)
				_, current_loss_D, current_loss_G = self.GenerationMethod.sess.run(
					[self.GenerationMethod.g_solver, self.GenerationMethod.D_loss, self.GenerationMethod.G_loss],
						feed_dict = {self.GenerationMethod.x: x_mb, 
									 self.GenerationMethod.z: z_mb, 
									 self.GenerationMethod.noise: n_mb, 
									 self.GenerationMethod.is_training: self.GenerationMethod.if_BNTrainingMode})

				train_loss_D += (current_loss_D * batch_size)
				train_loss_G += (current_loss_G * batch_size)
			
			train_loss_D /= n_train
			train_loss_G /= n_train 

			if use_validation:
				for _ in range(1, n_valid // batch_size + 1):
					x_mb = self.GenerationMethod.sample_data_np(valid_data, batch_size)
					z_mb = self.DisentangleMethod.encode(x_mb)
					n_mb = self.GenerationMethod.sample_z(batch_size, self.GenerationMethod.n_dim)

					current_loss_valid_D, current_loss_valid_G = self.GenerationMethod.sess.run(
						[self.GenerationMethod.D_loss, self.GenerationMethod.G_loss], 
							feed_dict = {self.GenerationMethod.x: x_mb, 
										 self.GenerationMethod.z: z_mb, 
										 self.GenerationMethod.noise: n_mb, self.GenerationMethod.is_training: False})

					valid_loss_D += current_loss_valid_D
					valid_loss_G += current_loss_valid_G
				
				valid_loss_D /= n_valid
				valid_loss_G /= n_valid
			
			self.GenerationMethod.train_loss_D.append(train_loss_D)
			self.GenerationMethod.train_loss_G.append(train_loss_G)
			self.GenerationMethod.valid_loss_D.append(valid_loss_D)
			self.GenerationMethod.valid_loss_G.append(valid_loss_G)
			self.GenerationMethod.training_time += (time.time() - begin)

			# testing for generation metrics
			if (epoch - 1) % test_every_n_epochs == 0 and use_test_during_train:
				
				if test_data is None:
					reset_test_data = True
					sampled_indices = sample(range(n_train), test_size)
					
					test_data = train_data[sampled_indices, :]
					test_cond =  self.DisentangleMethod.encode(test_data)

					gen_data = self.GenerationMethod.generate_cells(test_cond)

					if inception_score_data is not None:
						inception_score_subdata = inception_score_data[sampled_indices]
						mean_is_real, std_is_real = genmetric.InceptionScore(test_data, inception_score_subdata, test_data)
						mean_is_fake, std_is_fake = genmetric.InceptionScore(test_data, inception_score_subdata, gen_data)
					else:
						mean_is_real = std_is_real = mean_is_fake = std_is_fake = 0.0

				else:
					assert test_data.shape[0] == test_size
					reset_test_data = False

					test_cond =  self.DisentangleMethod.encode(test_data)
					gen_data = self.GenerationMethod.generate_cells(test_cond)

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
					test_cond = None

			if verbose: 
				print(f"Epoch {epoch}: D Train Loss: {train_loss_D} G Train Loss: {train_loss_G}  D Valid Loss: {valid_loss_D} G Valid Loss: {valid_loss_G}")

			# early stopping
			if use_validation and epoch > 1:
				if abs(self.GenerationMethod.valid_loss_D[epoch - 2] - self.GenerationMethod.valid_loss_D[epoch - 1]) > min_delta or abs(self.GenerationMethod.valid_loss_G[epoch - 2] - self.GenerationMethod.valid_loss_G[epoch - 1]) > min_delta:
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