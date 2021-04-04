#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from random import sample
import tensorflow as tf
from scipy import linalg
from tensorflow import distributions as ds


def permute_dims(z, opt):
	permuted_rows = []
	zuse = tf.identity(z, name="input")
	cat_part = zuse[:, :opt.non_noise_cat]
	cat_part_sf = tf.random_shuffle(cat_part)
	for i in range(zuse.get_shape()[1]):
		if i >= opt.non_noise_cat:
			permuted_rows.append(tf.random_shuffle(zuse[:, i]))
	permuted_samples = tf.stack(permuted_rows, axis=1)
	permuted_samples = tf.concat([cat_part_sf, permuted_samples], axis = 1)
	permuted_output = tf.identity(permuted_samples, name = "output")
	return permuted_output

def con_noise_prior(con_tensor, batch_size, dim):
	rez = np.zeros([batch_size, dim])
	for t in range(con_tensor.shape[1]):
		sam_index = sample(range(con_tensor.shape[0]), batch_size)
		input_realize = con_tensor[sam_index, t]
		rez[:, t] = input_realize
	if dim > con_tensor.shape[1]:
		rez[:, con_tensor.shape[1]:] = np.random.normal(0.0, scale = 1.0, size=(batch_size, dim - (con_tensor.shape[1])))
	return rez


def random_fix_prior(con_tensor, con_place, batch_size, dim, opt):
	rez = np.zeros([batch_size, dim])
	for t in range(con_tensor.shape[1]):
		sam_index = sample(range(con_tensor.shape[0]), batch_size)
		input_realize = con_tensor[sam_index, t]
		rez[:, t] = input_realize
	if dim > con_tensor.shape[1]:
		rez[:, con_tensor.shape[1]:] = np.random.normal(0.0, scale = 1.0, size=(batch_size, dim - (con_tensor.shape[1])))
	z_use = rez.copy()
	h_data = z_use[:, opt.non_noise_cat + con_place]
	
	rez_2 = np.zeros([batch_size, dim])
	for t in range(con_tensor.shape[1]):
		sam_index = sample(range(con_tensor.shape[0]), batch_size)
		input_realize = con_tensor[sam_index, t]
		rez_2[:, t] = input_realize
	if dim > con_tensor.shape[1]:
		rez_2[:, con_tensor.shape[1]:] = np.random.normal(0.0, scale = 1.0, size=(batch_size, dim - (con_tensor.shape[1])))
	rez_2[:, opt.non_noise_cat + con_place] = h_data
	return rez, rez_2


def fix_noise_prior(con_tensor, batch_size, dim):
	input_realize = con_tensor
	rez = np.zeros([batch_size, dim])
	if dim == (input_realize.shape[1]):
		rez = input_realize
	else:
		rez[:, :(input_realize.shape[1])] = input_realize
		rez[:, (input_realize.shape[1]):] = np.random.normal(0.0, scale = 1.0, size=(batch_size, dim - (input_realize.shape[1])))
	return rez

def noise_prior(batch_size, dim):
	temp_norm = np.random.normal(0.0, scale = 1.0, size=(batch_size, dim))
	return temp_norm

def prior(batch_size, dim):
	shp = [batch_size, dim]
	loc = tf.zeros(shp)
	scale = tf.ones(shp)
	return ds.Normal(loc, scale)

def random_uc(insize, opt):
	idxs = np.random.randint(opt.non_noise_cat, size = insize)
	onehot = np.zeros((insize, opt.non_noise_cat))
	onehot[np.arange(insize), idxs] = 1
	return onehot, idxs

def random_z(size, opt):
	rez = np.zeros([size, opt.noise_input_size])
	rez[:, :opt.non_noise_cat], idxs = random_uc(size, opt)
	rez[:, opt.non_noise_cat:] = noise_prior(size, opt.noise_input_size - opt.non_noise_cat)
	return rez, idxs

def random_fix_z(y_data, opt):
	rez = np.zeros([y_data.shape[0], opt.noise_input_size])
	rez[:, :opt.non_noise_cat] = y_data
	rez[:, opt.non_noise_cat:] = noise_prior(y_data.shape[0], opt.noise_input_size - opt.non_noise_cat)
	return rez

# con_place equal to 0 or 1 for 2 continuous variables of ground truth
def random_fix_z_con(size, con_place, opt):
	z_data, _ = random_z(size, opt)
	z_use = z_data.copy()
	h_data = z_use[:, opt.non_noise_cat + con_place]
	z_data_2, _ = random_z(size, opt)
	z_data_2[:, opt.non_noise_cat + con_place] = h_data
	return z_data, z_data_2	

def random_fix_noise_prior(size, con_place, opt):
	z_data = noise_prior(size, opt.noise_input_size_2)
	z_use = z_data.copy()
	h_data = z_use[:, opt.non_noise_cat + con_place]
	z_data_2 = noise_prior(size, opt.noise_input_size_2)
	z_data_2[:, opt.non_noise_cat + con_place] = h_data
	return z_data, z_data_2	

def log(x, opt):
	return tf.log(x + opt.epsilon_use)

def sample_X(X, size):
	start_idx = np.random.randint(0, X.shape[0] - size)
	return X[start_idx:start_idx + size, :]

def sample_XY(X, Y, size):
	start_idx = np.random.randint(0, X.shape[0] - size)
	return X[start_idx:start_idx + size, :], Y[start_idx:start_idx + size, :]


def preds2score(PYX, PY, eps = 1e-6, splits=3):
	scores = []
	for i in range(splits):
		part = PYX[(i * PYX.shape[0] // splits):((i + 1) * PYX.shape[0] // splits), :]
		part = part + eps
		kl = part * (np.log(part) - np.log(np.expand_dims(PY, 0)))
		kl = np.mean(np.sum(kl, 1))
		scores.append(np.exp(kl))
	return np.mean(scores), np.std(scores)


def generateTheta(L, ndim):
	# This function generates L random samples from the unit `ndim'-u
	theta=[w/np.sqrt((w**2).sum()) for w in np.random.normal(size=(L, ndim))]
	return np.asarray(theta)



def calculate_statistics(numpy_data):
	mu = np.mean(numpy_data, axis = 0)
	sigma = np.cov(numpy_data, rowvar = False)
	return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps = 1e-6):
	diff = mu1 - mu2
	covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp = False)
	if not np.isfinite(covmean).all():
		msg = (
			'fid calculation produces singular product; '
			'adding %s to diagonal of cov estimates' % eps
		)
		print(msg)
		offset = np.eye(sigma1.shape[0]) * eps
		covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
	
	if np.iscomplexobj(covmean):
		if not np.allclose(np.diagonal(covmean).imag, 0, atol = 1e-3):
			m = np.max(np.abs(covmean.imag))
			raise ValueError('Cell component {}'.format(m))
		covmean = covmean.real
	
	tr_covmean = np.trace(covmean)


	return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean )


def calculate_fid_score(data1, data2):
	m1, s1 = calculate_statistics(data1)
	m2, s2 = calculate_statistics(data2)
	fid_value = calculate_frechet_distance(m1, s1, m2, s2)
	return fid_value
