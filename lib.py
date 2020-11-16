"""
Library of functions
"""
import numpy as np
from random import sample
import tensorflow as tf
from tensorflow import distributions as ds


def noise_prior(batch_size, dim):
	"""
	generating random samples with dimension (batch_size, dim) from standard Gaussian distributions
	"""
	temp_norm = np.random.normal(0.0, scale = 1.0, size=(batch_size, dim))
	return temp_norm

def tf_standardGaussian_prior(batch_size, dim):
	"""
	TensorFlow standard Gaussian distributions
	"""
	shp = [batch_size, dim]
	loc = tf.zeros(shp)
	scale = tf.ones(shp)
	return ds.Normal(loc, scale)

def log(x, opt):
	"""
	numerically stable log-transformation
	"""
	return tf.log(x + opt.epsilon_use)

def sample_X(X, size):
	"""
	sampling from X tensor
	"""
	start_idx = np.random.randint(0, X.shape[0] - size)
	return X[start_idx:start_idx + size, :]

def sample_XY(X, Y, size):
	"""
	simutaneously sampling from both X and Y tensors
	"""
	start_idx = np.random.randint(0, X.shape[0] - size)
	return X[start_idx:start_idx + size, :], Y[start_idx:start_idx + size, :]

def compute_gp(x, x_gen, disc, opt):
	"""
	compute the gradient penalty of GANs
	"""
	epsilon_x = tf.random_uniform([], 0.0, 1.0)
	x_hat = x * epsilon_x + (1 - epsilon_x) * x_gen
	d_hat, _ = disc(x_hat, opt, reuse = True)
	gradients = tf.gradients(d_hat, x_hat)[0]
	slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices = [1]))
	gradient_penalty =  tf.reduce_mean((slopes - 1.0) ** 2)
	return gradient_penalty

def compute_con_gp(x, x_gen, y, disc, opt):
	"""
	compute the gradient penalty of conditional GANs
	"""
	epsilon_x = tf.random_uniform([], 0.0, 1.0)
	x_hat = x * epsilon_x + (1 - epsilon_x) * x_gen
	d_hat, _ = disc(x_hat, y, opt, reuse=True)
	gradients = tf.gradients(d_hat, x_hat)[0]
	slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
	gradient_penalty =  tf.reduce_mean((slopes - 1.0) ** 2)
	return gradient_penalty

def MutualInformationLowerBound(c_rec, c_sample, opt):
	"""
	compute the mutual information lower bound for InfoGANs
	"""

	ll_con = None
	est_vec = c_rec[:, :opt.code_size]
	c_sample_vec = c_sample[:, :opt.code_size]

	if opt.InfoGAN_fix_std:
		std_vec = tf.ones_like(est_vec)
	else:
		std_vec = c_rec[:, opt.code_size: 2 * opt.code_size]
		std_vec = tf.nn.softplus(std_vec)

	ll_con_dist = ds.Normal(est_vec, std_vec)
	ll_conLogProb = ll_con_dist.log_prob(c_sample_vec)
	ll_con = tf.reduce_sum(ll_conLogProb, [1])
	
	result_con = tf.reduce_mean(ll_con)
	
	return result_con


def c_mutual_mu_var_entropy(z_gen_mean_v, z_gen_std_v, c_rec_sample, opt):
	"""
	evaluate the probabilities of samples given posterior Gaussian means and standard deviations
	"""

	con_real = c_rec_sample[:, :opt.code_size]

	z_norm = (con_real - z_gen_mean_v) / z_gen_std_v
	z_var = tf.square(z_gen_std_v)
	con_gen = -0.5 * (z_norm * z_norm + tf.log(z_var) + np.log(2 * np.pi))

	return con_gen

def logsumexp(value, opt,  dim = None, keepdims = False):
	"""
	Numerically stable calculation of log(sum(exp(...)))
	"""
	if dim is not None:
		m = tf.reduce_max(value, axis = dim, keepdims = True)
		value0 = tf.subtract(value, m)
		if keepdims is False:
			m = tf.squeeze(m, dim)
		return tf.add(m, log(tf.reduce_sum(tf.exp(value0), axis = dim, keepdims = keepdims), opt))

	else:
		m = tf.reduce_max(value)
		sum_exp = tf.reduce_sum(tf.exp(tf.subtract(value, m)))	
		return tf.add(m, log(sum_exp))


def estimate_minibatch_mss_entropy(z_gen_mean_v, z_gen_std_v, c_rec_sample, opt):
	"""
	estimate the approximated latent entropies based on the Minibatch Stratified Sampling (MSS) method
	"""
	dataset_size = tf.convert_to_tensor(opt.num_cells_train)
	# compute the weights
	output = tf.zeros((tf.shape(c_rec_sample)[0] - 1, 1))
	output = tf.concat([tf.ones((1, 1)), output], axis = 0)
	outpart_1 = tf.zeros((tf.shape(c_rec_sample)[0], 1))
	outpart_3 = tf.zeros((tf.shape(c_rec_sample)[0], tf.shape(c_rec_sample)[0] - 2))
	output = tf.concat([outpart_1, output], axis = 1)
	part_4 = - tf.concat([output, outpart_3], axis = 1)/tf.to_float(dataset_size)

	part_1 = tf.ones((tf.shape(c_rec_sample)[0], tf.shape(c_rec_sample)[0]))/tf.to_float(tf.shape(c_rec_sample)[0] - 1)
	part_2 = tf.ones((tf.shape(c_rec_sample)[0], tf.shape(c_rec_sample)[0]))
	part_2 = - tf.matrix_band_part(part_2, 1, 0)/tf.to_float(dataset_size)
	part_3 = tf.eye(tf.shape(c_rec_sample)[0]) * (2/tf.to_float(dataset_size) - 1/tf.to_float(tf.shape(c_rec_sample)[0] - 1))

	weights =  log(part_1 + part_2 + part_3 + part_4, opt)

	function_to_map = lambda x: c_mutual_mu_var_entropy(z_gen_mean_v, z_gen_std_v, tf.reshape(x, [1, opt.code_size]), opt) 
	logqz_i_m = tf.map_fn(function_to_map, c_rec_sample, dtype = tf.float32)
	weights_expand =  tf.expand_dims(weights, 2)
	logqz_i_margin = logsumexp(logqz_i_m + weights_expand, opt, dim = 1, keepdims = False)
	logqz_value = tf.reduce_sum(logqz_i_m, axis = 2, keepdims = False)
	logqz_v_joint = logsumexp(logqz_value + weights, opt, dim = 1, keepdims = False)
	logqz_sum = logqz_v_joint
	logqz_i_sum = logqz_i_margin

	marginal_entropies = (- tf.reduce_mean(logqz_i_sum, axis = 0))
	joint_entropies = (- tf.reduce_mean(logqz_sum)) 

	return marginal_entropies, joint_entropies

def estimate_entropy(z_gen_mean_v, z_gen_std_v, c_rec_sample, opt):
	"""
	estimate the latent entropies based on large-sample approximations
	"""
	weights = - log(tf.to_float(tf.shape(z_gen_mean_v)[0]), opt)

	function_to_map = lambda x: c_mutual_mu_var_entropy(z_gen_mean_v, z_gen_std_v, tf.reshape(x, [1, opt.code_size]), opt) 
	logqz_i_m = tf.map_fn(function_to_map, c_rec_sample, dtype = tf.float32)
	logqz_i_margin = logsumexp(logqz_i_m + weights, opt, dim = 1, keepdims = False)
	logqz_value = tf.reduce_sum(logqz_i_m, axis = 2, keepdims = False)
	logqz_v_joint = logsumexp(logqz_value + weights, opt, dim = 1, keepdims = False)
	logqz_sum = logqz_v_joint
	logqz_i_sum = logqz_i_margin

	marginal_entropies = (- tf.reduce_mean(logqz_i_sum, axis = 0))
	joint_entropies = (- tf.reduce_mean(logqz_sum)) 

	return marginal_entropies, joint_entropies