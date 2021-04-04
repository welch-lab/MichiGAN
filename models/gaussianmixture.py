#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
from sklearn import mixture

class GaussianMixtureModel:
	"""
	Gaussian mixture model 
	"""
	def __init__(self):
		super().__init__()

	def GMModel(self, n_components, covariance_type):
		"""
		sklearn GMM
		"""
		gmm = mixture.GaussianMixture(n_components = n_components, 
			covariance_type = covariance_type)

		return gmm

	def SelectTrain(self, train_data, use_validation = False, valid_data = None, 
		n_components_range = range(1, 101), 
		cv_types = ['spherical', 'tied', 'diag', 'full']):
		"""
		cross validation to select n_components and covariance_type
		"""
		bic_train = {}
		bic_valid = {}

		for covariance_type in cv_types:
			
			bic_train[covariance_type] = []
			bic_valid[covariance_type] = []

			for n_components in n_components_range:
			
				gmm = self.GMModel(n_components, covariance_type)
				
				try:
					gmm.fit(train_data)
					bic_train[covariance_type].append(gmm.bic(train_data))
					
					if use_validation:
						bic_valid[covariance_type].append(gmm.bic(valid_data))

				except:
					bic_train[covariance_type].append(0)

					if use_validation:
						bic_valid[covariance_type].append(0)
				

				

		train_loss_df = pd.DataFrame(bic_train)
		valid_loss_df = pd.DataFrame(bic_valid)

		return train_loss_df, valid_loss_df


	def fit(self, train_data, n_components, covariance_type, use_validation = False, valid_data = None):
		"""
		fit GMM model with training data
		"""
		gmm = self.GMModel(n_components, covariance_type)
		gmm.fit(train_data)

		
		train_loss = gmm.bic(train_data)
		valid_loss = 0.0
		if use_validation:
			valid_loss = gmm.bic(valid_data)

		return gmm, train_loss, valid_loss

	def reconstruct(self, gmm, x_data):
		"""
		generate data from GMM
		"""

		x_rec_data, rec_label = gmm.sample(x_data.shape[0])

		return x_rec_data, rec_label


