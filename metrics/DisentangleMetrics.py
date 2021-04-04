#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from random import sample

import scipy
from scipy import stats
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import tensorflow as tf
from tensorflow import distributions as ds

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class DisentangledRepresentation:
	"""
	Disentanglement performance class
	"""

	def __init__(self):
		super().__init__()
		self.clf = SVC(kernel = 'linear')


	def KS_test(self, z_sample_data, z_sample_data_2 = None):
		"""
		Kolmogorov-Smirnov test between two latent tensors on each dimension
		"""

		p_value_list = []
		if z_sample_data_2 is None:
			norm_data = np.random.normal(0, 1, size = z_sample_data.shape[0])
			for i in range(z_sample_data.shape[1]):
				stat, p_value = stats.ks_2samp(z_sample_data[:, i], norm_data)
				p_value_list.append(p_value)
		else:
			assert z_sample_data.shape[1] == z_sample_data_2.shape[1]
			for i in range(z_sample_data.shape[1]):
				stat, p_value = stats.ks_2samp(z_sample_data[:, i], z_sample_data_2[:, i])
				p_value_list.append(p_value)
		
		return p_value_list

	def CorrValue(self, z_data, GroundTruthVar, is_spearman = True):
		"""
		correlation between each latent dimension of z_data and GroundTruthVar
		"""

		cor_value = np.zeros((z_data.shape[1]))
		if is_spearman:
			for t in range(z_data.shape[1]):
				cor_value[t] = scipy.stats.spearmanr(z_data[:, t], GroundTruthVar, nan_policy = 'omit')[0]
		else:
			for t in range(z_data.shape[1]):
				cor_value[t] = scipy.stats.pearsonr(z_data[:, t], GroundTruthVar, nan_policy = 'omit')[0]
		return cor_value

	def CorrGap(self, cor_matrix):
		"""
		correlation gap based on a correlation matrix
		"""

		cor_output = cor_matrix.copy()
		cor_output = np.abs(cor_output)
		
		cor_output.sort(axis = 1)
		metric = np.mean(cor_output[:, -1] - cor_output[:, -2])
		
		return metric


	def DataByCode(self, umap_data, z_data, path_figure_save):
		"""
		UMAP plots of data colored by each dimension of latent z_data
		"""
		
		dict_use = {}
		for h in range(z_data.shape[1]):
			dict_use["Z" + str(h+1)] = h + 1

		min_x, min_y = np.floor(umap_data['x-umap'].min()), np.floor(umap_data['y-umap'].min())
		max_x, max_y = np.ceil(umap_data['x-umap'].max()), np.ceil(umap_data['y-umap'].max())

		newfig = plt.figure(figsize=[20,6])
		for m in range(len(dict_use)):
			name_i = list(dict_use.keys())[m]
			num_i = dict_use[name_i]
			ax1 = newfig.add_subplot(2, 5, num_i)
			cb1 = ax1.scatter(umap_data['x-umap'], umap_data['y-umap'], s= 1, c = z_data[:, m], cmap= "plasma")
			ax1.tick_params(axis='x',          # changes apply to the x-axis
							which='both',      # both major and minor ticks are affected
							bottom=False,      # ticks along the bottom edge are off
							top=False,         # ticks along the top edge are off
							labelbottom=False) # labels along the bottom edge are off
			ax1.tick_params(axis='y',          # changes apply to the x-axis
							which='both',      # both major and minor ticks are affected
							bottom=False,      # ticks along the bottom edge are off
							top=False,         # ticks along the top edge are off
							labelbottom=False) # labels along the bottom edge are off
			ax1.get_yaxis().set_ticks([])
			ax1.set_ylim(min_y, max_y)
			ax1.set_xlim(min_x, max_x)
			ax1.set_title(name_i)

		newfig.savefig(path_figure_save, dpi=300, useDingbats = False)


	def MutualInformation(self, data, metadata, dictmeta, list_GroundTruthVars, z_dim, 
						  sess, MarginalEntropy, X_v, if_VarBalanced = True):
		"""
		mutual information for PCA. use self.mu and self.pca_posterior_mean(X) as X_v and data
		"""

		cond_en = np.zeros((len(list_GroundTruthVars),  z_dim))

		if data.shape[0] <= 12000:
			input_mig, meta_mig = data, metadata
		else:
			idx_mig = sample(range(data.shape[0]), 12000)
			input_mig = data[idx_mig, :]
			meta_mig = metadata.iloc[idx_mig, :]
		
		mar_en_v = sess.run(MarginalEntropy, {X_v: input_mig})

		# with several ground truth variables
		for k in range(len(list_GroundTruthVars)):
			k_var = list_GroundTruthVars[k]
			list_k = dictmeta[k_var]

			for k_use in list_k:
				value_list_k = np.where(np.array(meta_mig[k_var]) == k_use)[0]
				x_value_list_k = input_mig[value_list_k, :]
				en_i = sess.run(MarginalEntropy, {X_v: x_value_list_k})
				cond_en[k, :] += en_i * float(value_list_k.shape[0])/float(meta_mig.shape[0])

		if if_VarBalanced:
			factor_entropies = np.log([ len(dictmeta[i]) for i in list_GroundTruthVars])
		else:
			factor_entropies = np.array([scipy.stats.entropy(meta_mig[i].value_counts()) for i in list_GroundTruthVars])

		MIGValue, NormMutualInfo = self.MIGMetric(mar_en_v, cond_en, factor_entropies)
		
		return MIGValue, NormMutualInfo

	def MutualInformationWithMissing(self, data, metadata, dictmeta, list_GroundTruthVars, z_dim, 
									 sess, MarginalEntropy, X_v, training = None, if_VarBalanced = True):
		"""
		general mutual information for PCA or VAE on data with missing metadata
		"""
		cond_en = np.zeros((len(list_GroundTruthVars),  z_dim))
		mar_en = np.zeros((len(list_GroundTruthVars),  z_dim))

		if data.shape[0] <= 12000:
			input_mig, meta_mig = data, metadata
		else:
			idx_mig = sample(range(data.shape[0]), 12000)
			input_mig = data[idx_mig, :]
			meta_mig = metadata.iloc[idx_mig, :]
		
		for k in range(len(list_GroundTruthVars)):
			k_var = list_GroundTruthVars[k]
			input_obs = input_mig[-meta_mig[k_var].isna()]
			if training is None:
				mar_en[k, :] = sess.run(MarginalEntropy, {X_v: input_obs})
			else:
				mar_en[k, :] = sess.run(MarginalEntropy, {X_v: input_obs, training: False})

		# with several ground truth variables
		for k in range(len(list_GroundTruthVars)):
			k_var = list_GroundTruthVars[k]
			list_k = dictmeta[k_var]

			for k_use in list_k:
				value_list_k = np.where(np.array(meta_mig[k_var]) == k_use)[0]
				x_value_list_k = input_mig[value_list_k, :]

				if training is None:
					en_i = sess.run(MarginalEntropy, {X_v: x_value_list_k})
				else:
					en_i = sess.run(MarginalEntropy, {X_v: x_value_list_k, training: False})

				cond_en[k, :] += en_i /float(len(list_k))

		if if_VarBalanced:
			factor_entropies = np.log([ len(dictmeta[i]) for i in list_GroundTruthVars])
		else:
			factor_entropies = np.array([scipy.stats.entropy(meta_mig[i].value_counts()) for i in list_GroundTruthVars])

		MutualInfo = mar_en - cond_en
		mi_normed = MutualInfo/factor_entropies[:, None]
		mi_output = mi_normed.copy()
		
		mi_normed.sort(axis = 1)
		metric = np.mean(mi_normed[:, -1] - mi_normed[:, -2])
		
		return metric, mi_output
	
	def MutualInformationVAE(self, data, metadata, dictmeta, list_GroundTruthVars, z_dim, 
							 sess, MarginalEntropy, X_v, training, if_VarBalanced = True):
		"""
		mutual information for VAEs
		"""

		cond_en = np.zeros((len(list_GroundTruthVars),  z_dim))

		if data.shape[0] <= 12000:
			input_mig, meta_mig = data, metadata
		else:
			idx_mig = sample(range(data.shape[0]), 12000)
			input_mig = data[idx_mig, :]
			meta_mig = metadata.iloc[idx_mig, :]
		
		mar_en_v = sess.run(MarginalEntropy, {X_v: input_mig, training: False})

		# with several ground truth variables
		for k in range(len(list_GroundTruthVars)):
			k_var = list_GroundTruthVars[k]
			list_k = dictmeta[k_var]

			for k_use in list_k:
				value_list_k = np.where(np.array(meta_mig[k_var]) == k_use)[0]
				x_value_list_k = input_mig[value_list_k, :]
				en_i = sess.run(MarginalEntropy, {X_v: x_value_list_k, training: False})
				cond_en[k, :] += en_i * float(value_list_k.shape[0])/float(meta_mig.shape[0])

		if if_VarBalanced:
			factor_entropies = np.log([ len(dictmeta[i]) for i in list_GroundTruthVars])
		else:
			factor_entropies = np.array([scipy.stats.entropy(meta_mig[i].value_counts()) for i in list_GroundTruthVars])

		MIGValue, NormMutualInfo = self.MIGMetric(mar_en_v, cond_en, factor_entropies)
		
		return MIGValue, NormMutualInfo	


	def MIGMetric(self, marginal_entropies, con_entropies, factor_entropies):
		"""
		calculate mutual information gap (MIG) based on marginal and conditional entropies
		"""
		MutualInfo = marginal_entropies[None] - con_entropies
		mi_normed = MutualInfo/factor_entropies[:, None]
		mi_output = mi_normed.copy()
		
		mi_normed.sort(axis = 1)
		metric = np.mean(mi_normed[:, -1] - mi_normed[:, -2])
		
		return metric, mi_output

	def PlotBarCor(self, cor_matrix, path_figure_save = None):
		"""
		Plot correlation bar plots
		"""
		# Correlation GAP 
		rep_list = ["Z" + str(i + 1) for i in range(cor_matrix.shape[1])]
		newfig = plt.figure(figsize=[8,6])
		ax1 = newfig.add_subplot(2, 2, 1)
		cb1 = ax1.bar(rep_list, cor_matrix[0, :],  color= "blue")

		ax1.tick_params(
			axis='x',          # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom=False,      # ticks along the bottom edge are off
			top=False,         # ticks along the top edge are off
			labelbottom=False) # labels along the bottom edge are off
		ax1.set_ylim(-1, 1)
		ax1.set_title("Batch", fontsize = 18)

		ax1 = newfig.add_subplot(2,2, 2)
		cb1 = ax1.bar(rep_list, cor_matrix[1, :],  color= "blue")

		ax1.tick_params(
			axis='x',          # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom=False,      # ticks along the bottom edge are off
			top=False,         # ticks along the top edge are off
			labelbottom=False) # labels along the bottom edge are off
		ax1.set_ylim(-1, 1)
		ax1.set_title("Path", fontsize = 18)

		ax1 = newfig.add_subplot(2,2, 3)
		cb1 = ax1.bar(rep_list, cor_matrix[2, :],  color= "blue")

		ax1.tick_params(
			axis='x',          # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom=False,      # ticks along the bottom edge are off
			top=False,         # ticks along the top edge are off
			labelbottom=False) # labels along the bottom edge are off
		ax1.set_ylim(-1, 1)
		ax1.set_title("Step", fontsize = 18)

		ax1 = newfig.add_subplot(2,2, 4)
		cb1 = ax1.bar(rep_list, cor_matrix[3, :],  color= "blue")

		ax1.tick_params(
			axis='x',          # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom=False,      # ticks along the bottom edge are off
			top=False,         # ticks along the top edge are off
			labelbottom=False) # labels along the bottom edge are off
		ax1.set_ylim(-1, 1)
		ax1.set_title("Library Size Quartile", fontsize = 18)

		newfig.text(0.5, 0.04, 'Representations', ha='center', fontsize = 18)
		newfig.text(0.04, 0.5, 'Spearman Correlation', va='center', rotation='vertical', fontsize = 18)

		if path_figure_save is not None:
			newfig.savefig(path_figure_save, dpi=300, useDingbats = False)

	def PlotBarMI(self, norm_mi, path_figure_save = None):
		"""
		Plot normalized mutual information bar plots
		"""
		# normalized mutual information
		rep_list = ["Z" + str(i + 1) for i in range(norm_mi.shape[1])]

		newfig = plt.figure(figsize=[8,6])

		ax1 = newfig.add_subplot(2, 2, 1)
		cb1 = ax1.bar(rep_list, norm_mi[0, :],  color= "blue")

		ax1.tick_params(
			axis='x',          # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom=False,      # ticks along the bottom edge are off
			top=False,         # ticks along the top edge are off
			labelbottom=False) # labels along the bottom edge are off
		ax1.set_ylim(0, 1)
		ax1.set_title("Batch", fontsize = 18)

		ax1 = newfig.add_subplot(2,2, 2)
		cb1 = ax1.bar(rep_list, norm_mi[1, :],  color= "blue")

		ax1.tick_params(
			axis='x',          # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom=False,      # ticks along the bottom edge are off
			top=False,         # ticks along the top edge are off
			labelbottom=False) # labels along the bottom edge are off
		ax1.set_ylim(0, 1)
		ax1.set_title("Path", fontsize = 18)

		ax1 = newfig.add_subplot(2,2, 3)
		cb1 = ax1.bar(rep_list, norm_mi[2, :],  color= "blue")

		ax1.tick_params(
			axis='x',          # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom=False,      # ticks along the bottom edge are off
			top=False,         # ticks along the top edge are off
			labelbottom=False) # labels along the bottom edge are off
		ax1.set_ylim(0, 1)
		ax1.set_title("Step", fontsize = 18)

		ax1 = newfig.add_subplot(2,2, 4)
		cb1 = ax1.bar(rep_list, norm_mi[3, :],  color= "blue")

		ax1.tick_params(
			axis='x',          # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom=False,      # ticks along the bottom edge are off
			top=False,         # ticks along the top edge are off
			labelbottom=False) # labels along the bottom edge are off
		ax1.set_ylim(0, 1)
		ax1.set_title("Library Size Quartile", fontsize = 18)


		newfig.text(0.5, 0.04, 'Representations', ha='center', fontsize = 18)
		newfig.text(0.04, 0.5, 'Normalized Mutual Information', va='center', rotation='vertical', fontsize = 18)

		if path_figure_save is not None:
			newfig.savefig(path_figure_save, dpi=300, useDingbats = False)


	def latent_space_entropies(self, tf_JointEntropy, tf_sess, tf_z_mu, tf_z_std, tf_z_sample, tf_training, 
							   z_mu, z_std, z_sample):
		"""
		calculate latent space entropy based on latent values and given posterior distributions
		"""
		feed_dict = {
			tf_z_mu: z_mu, tf_z_std: z_std, 
			tf_z_sample: z_sample, tf_training: False
			}

		z_JointEntropy = tf_sess.run(tf_JointEntropy, feed_dict = feed_dict)

		return z_JointEntropy

	def FactorVAEMetric(self, input_data, data_meta, list_GroundTruthVars, dict_meta, latent_dim, input_latent, 
						K_samples = 10000, L_samples = 40):
		"""
		calculate the disentanglement metrics of beta-VAE and FactorVAE
		"""
		k_list = []

		mean_list, std_list = None, None

		all_c = input_latent
		# put generated c information into its standard deviation
		all_c_std = all_c.std(axis = 0, keepdims = True)

		for s in range(K_samples):

			k = sample(range(len(list_GroundTruthVars)), 1)
			
			# FactorVAE metric
			fixed_factor = list_GroundTruthVars[k[0]]
			fixed_factor_value_list = dict_meta[fixed_factor]
			fixed_factor_value_kim = sample(fixed_factor_value_list, 1)[0]
			# the data with this fixed value
			list_fixed_value_kim = np.where(data_meta[fixed_factor] == fixed_factor_value_kim)[0].tolist()

			# sample without replacement for FactorVAE metric
			indexkim = np.random.choice(list_fixed_value_kim , L_samples, replace = False)
			selectkimdata = input_data[indexkim, :]

			ckim = input_latent[indexkim, :]
			ckim_scale = ckim/all_c_std
			
			diff_list = None    

			# beta-vae metric
			for l in range(L_samples):

				fixed_factor_value = sample(fixed_factor_value_list, 1)[0]
				list_fixed_value = np.where(data_meta[fixed_factor] == fixed_factor_value)[0].tolist()

				indexbeta = np.random.choice(list_fixed_value, 2, replace = False)

				z1 = input_latent[[indexbeta[0]], :]
				z2 = input_latent[[indexbeta[0]], :]
				
				# for the categorial variable from onehot to categories
				if diff_list is None:
					diff_list =  np.abs(z1 - z2)
				else:
					diff_list = np.append(diff_list, np.abs(z1 - z2), axis = 0)

			mean_diff = np.mean(diff_list, axis = 0, keepdims = 1)
			mean_diff_subdim = mean_diff[:, latent_dim]

			std_diff = np.std(ckim_scale, axis = 0, keepdims = 1)
			std_diff_subdim = std_diff[:, latent_dim]
			std_diff_max = np.argmax(std_diff_subdim)

			if mean_list is None:
				mean_list = mean_diff_subdim
			else:
				mean_list = np.append(mean_list, mean_diff_subdim, axis = 0)
			
			if std_list is None:
				std_list = std_diff_max
			else:
				std_list = np.append(std_list, std_diff_max)
			
			k_list.append(k[0])

		# classifier (cross-validation)
		train_index = range(int(K_samples * 0.8))
		test_index = range(int(K_samples * 0.8), K_samples)
		
		#k_list_n = [j for sub in k_list for j in sub] # if [[0], [1], ...] is used for k_list
		X_train, X_test = mean_list[train_index, :], mean_list[test_index, :]
		y_train, y_test = np.array(k_list)[train_index], np.array(k_list)[test_index]
		
		# SVM classifier
		self.clf.fit(X_train, y_train)
		predictions_train = self.clf.predict(X_train)
		predictions_test = self.clf.predict(X_test)

		betaVAE_train = np.mean((predictions_train == y_train)*1)
		betaVAE_test = np.mean((predictions_test == y_test)*1)

		#  FactorVAE
		X_train, X_test = std_list[train_index], std_list[test_index]
		y_train, y_test = np.array(k_list)[train_index], np.array(k_list)[test_index]
		
		# majority vote classifier
		X_list, y_list = np.unique(X_train), np.unique(y_train)
		v = np.zeros((len(X_list), len(y_list)))
		for ind in range(len(X_train)):
			row, col = X_train[ind], y_train[ind]
			row_in, col_in = list(np.where(row == X_list)[0])[0], list(np.where(col == y_list)[0])[0]
			v[row_in, col_in] += 1
		pre_model = np.argmax(v, axis = 1)

		# majority vote predictions
		predictions_beta_train = np.zeros(y_train.shape)
		predictions_beta_test = np.zeros(y_test.shape)

		for i in range(len(X_train)):
			t = X_train[i]
			if list(np.where(t == X_list)[0]) == []:
				predictions_beta_train[i] = 1000 # an arbitrary large category
			else:
				predictions_beta_train[i] = pre_model[list(np.where(t == X_list)[0])[0]]

		factorVAE_train = np.mean((predictions_beta_train == y_train)*1)

		
		for i in range(len(X_test)):
			t = X_test[i]
			if list(np.where(t == X_list)[0]) == []:
				predictions_beta_test[i] = 1000 # an arbitrary large category
			else:
				predictions_beta_test[i] = pre_model[list(np.where(t == X_list)[0])[0]]

		factorVAE_test = np.mean((predictions_beta_test == y_test)*1)

		return betaVAE_train, betaVAE_test, factorVAE_train, factorVAE_test

	def KNNPredictVar(self, pca_real, pca_fake, GroundTruthVar_real):
		"""
		predict the ground-truth varable values of fake data based on the k-nearest neighbor algorithm
		trained on real PC values. 
		"""
		
		neigh = KNeighborsClassifier(n_neighbors = 3)
		neigh.fit(pca_real, GroundTruthVar_real)
		
		GroundTruthVar_fake = neigh.predict(pca_fake)

		return GroundTruthVar_fake



class LatentSpaceVectorArithmetic:
	""" 
	Latent space vector arithmetic algorithm  
	"""
	def __init__(self):
		super().__init__()

	def Union(self, lst1, lst2):
		"""
		union of two lists
		""" 
		final_list = list(set(lst1) | set(lst2)) 
		return final_list 

	def balancer(self, data, meta_data, consider_trt):
		"""
		balance data based on treatment
		"""

		list_trt = list(meta_data['treatment'])
		class_pop = {}
		for cls in consider_trt:
			class_pop[cls] = meta_data.copy()[meta_data['treatment'] == cls].shape[0]

		max_number = np.max(list(class_pop.values()))
		all_data = None
		meta_all = None

		for cls in consider_trt:
			idx = [i for i in range(len(list_trt)) if list_trt[i] == cls]
			
			temp = data.copy()[idx, :]
			temp_meta = meta_data.copy().iloc[idx, :]

			index = np.random.choice(range(temp.shape[0]), max_number)
			temp_x = temp[index, :]
			meta_x = temp_meta.iloc[index, :]
			if all_data is None:
				all_data = temp_x
				meta_all = meta_x
			else:
				all_data = np.concatenate((all_data, temp_x), axis = 0)
				meta_all = pd.concat([meta_all, meta_x])

		return all_data, meta_all

	def sample_for_effect(self, data_1, data_2, meta_1, meta_2, consider_trt):
		"""
		give two datasets with different cell types, 
		balance each dataset by treatment type
		balance two datasets in size
		"""
		data_1_b, meta_1_b = self.balancer(data_1, meta_1, consider_trt)
		data_2_b, meta_2_b = self.balancer(data_2, meta_2, consider_trt)

		balance_size = min(data_1_b.shape[0], data_2_b.shape[0])
		idx_1 = np.random.choice(range(data_1_b.shape[0]), size = balance_size, replace = False)
		idx_2 = np.random.choice(range(data_2_b.shape[0]), size = balance_size, replace = False)

		data_use1, data_use2 = data_1_b[idx_1, :], data_2_b[idx_2, :]
		return data_use1, data_use2


	def latent_vector_arithmetic(self, tf_sess, tf_z_latent, tf_x_data, tf_training, x_data1, x_data2):
		"""
		calculate the averaged latent difference between two samples
		"""
		
		feed_dict1 = {tf_x_data: x_data1, tf_training: False}
		z_latent1 = tf_sess.run(tf_z_latent, feed_dict = feed_dict1)

		feed_dict2 = {tf_x_data: x_data2, tf_training: False}
		z_latent2 = tf_sess.run(tf_z_latent, feed_dict = feed_dict2)

		z_diff = z_latent1.mean(0) - z_latent2(0)

		return z_diff

	def generate_from_latent(self, tf_sess, tf_z_latent, tf_x_rec_data, tf_training, z_data):
		"""
		generate data from latent samples
		"""

		feed_dict = {tf_z_latent: z_data, tf_training: False}
		
		return tf_sess.run(tf_x_rec_data, feed_dict = feed_dict)

	def estimate(self, x_data1, x_data2, meta_1, meta_2, consider_trt, data_true, data_control, 
				 tf_sess, tf_z_latent, tf_x_rec_data, tf_x_data, tf_training):
		"""
		latent space vector arithmetic
		1 is for control, 
		2 is for target 
		"""

		x_data1, x_data2 = self.sample_for_effect(x_data1, x_data2, meta_1, meta_2, consider_trt)
		z_diff = self.latent_vector_arithmetic(tf_sess, tf_z_latent, tf_x_data, x_data1, x_data2)

		n_true, n_control = data_true.shape[0], data_control.shape[0]

		# make sure the predicted data, true data and control data all have the same size
		if n_true < n_control:

			sample_indices = sample(range(n_control), n_true)
			input_control = data_control[sample_indices]

		elif n_true > n_control:

			sample_indices = np.random.choice(n_control, n_true, replace = True)
			input_control = data_control[sample_indices]
			
		else:

			input_control = data_control

		feed_dict = {tf_x_data: input_control, tf_training: False}
		z_control = tf_sess.run(tf_z_latent, feed_dict = feed_dict)

		z_pred = z_control - z_diff
		x_pred_data = self.generate_from_latent(tf_sess, tf_z_latent, tf_x_rec_data, 
			z_pred)

		return x_pred_data, input_control