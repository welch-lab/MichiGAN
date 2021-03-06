"""
Code for training beta-TCVAE (beta = 100) on Tabula Muris heart data
"""
import os
os.chdir('..')

import math

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import distributions as ds


from nets import *
from lib import *


class Options(object):
	def __init__(self, num_cells_train, gex_size):
		self.num_cells_train =  num_cells_train  # number of cells
		self.gex_size = gex_size                 # number of genes
		self.epsilon_use = 1e-16                 # small constant value
		self.n_train_epochs = 100                 # number of epochs for training
		self.batch_size = 32                     # batch size of the GAN-based methods
		self.vae_batch_size = 128                # batch size of the VAE-based methods
		self.code_size = 10                      # number of codes
		self.noise_size = 118                    # number of noise variables
		self.inflate_to_size1 = 256              # number of neurons
		self.inflate_to_size2 = 512              # number of neurons
		self.inflate_to_size3 = 1024             # number of neurons
		self.TotalCorrelation_lamb = 100.0         # hyperparameter for the total correlation penalty in beta-TCVAE
		self.InfoGAN_fix_std = True              # fixing the standard deviation or not for the Q network of InfoGAN
		self.dropout_rate = 0.2                  # dropout hyperparameter
		self.disc_internal_size1 = 1024          # number of neurons
		self.disc_internal_size2 = 512           # number of neurons
		self.disc_internal_size3 = 10            # number of neurons
		self.num_cells_generate = 3000           # number of sampled cells
		self.GradientPenaly_lambda = 10.0        # hyperparameter for the gradient penalty of Wasserstein GANs
		self.latentSample_size = 1               # number of samples of the encoder of VAEs
		self.MutualInformation_lamb = 10.0       # hyperparameter for the mutual information penalty in InfoGAN
		self.Diters = 5                          # number of training discriminator network per training of generator network of Wasserstein GANs
		self.model_path = "./examples/models_tcvae/"        # path saving the model

#################################################################################
# load data and define hyperparameters
#################################################################################
data_matrix = np.load('./data/TabulaMurisHeart_Processed.npy')
data_meta = pd.read_csv("./data/TabulaMurisHeart_MetaInformation.csv")
opt = Options(data_matrix.shape[0], data_matrix.shape[1])

#################################################################################
# define network tensors
#################################################################################
z_v = tf.placeholder(tf.float32, shape = (None, opt.code_size))
X_v = tf.placeholder(tf.float32, shape = (None, opt.gex_size))

## encoder
z_gen_mean_v, z_gen_std_v = vaes_encoder(X_v, opt)

### reparameterization of latent space
batch_size = tf.shape(z_gen_mean_v)[0]
eps = tf.random_normal(shape=[batch_size, opt.code_size])
z_gen_data_v = z_gen_mean_v + z_gen_std_v * eps

### latent entropies in a minibatch
margin_entropy_mss, joint_entropy_mss = estimate_minibatch_mss_entropy(z_gen_mean_v, z_gen_std_v, z_gen_data_v, opt)
### total correlation in a minibatch
TotalCorre_mss = tf.reduce_sum(margin_entropy_mss) - tf.reduce_sum(joint_entropy_mss)

## decoder
z_gen_decoder = vaes_decoder(z_gen_data_v, opt)

### generated data
X_gen_data = z_gen_decoder.sample(opt.latentSample_size)
X_gen_data = tf.reshape(X_gen_data , tf.shape(X_gen_data)[1:])

## loss elements
### reconstruction error
z_gen_de = z_gen_decoder.log_prob(X_v)
z_gen_de_value = tf.reduce_sum(z_gen_de, [1])
rec_x_loss = - tf.reduce_mean(z_gen_de_value)

### latent prior and posterior probabilities
stg_prior = tf_standardGaussian_prior(tf.shape(X_v)[0], opt.code_size)
latent_prior = stg_prior.log_prob(z_gen_data_v)
latent_posterior = c_mutual_mu_var_entropy(z_gen_mean_v, z_gen_std_v, z_gen_data_v, opt)

### latent joint prior and posterior probabilities
latent_prior_joint = tf.reduce_sum(latent_prior, [1])
latent_posterior_joint = tf.reduce_sum(latent_posterior, [1])

### KL divergence
kl_latent = - tf.reduce_mean(latent_prior_joint) + tf.reduce_mean(latent_posterior_joint)

### VAE/beta-TCVAE loss function
obj_vae = rec_x_loss  + kl_latent + opt.TotalCorrelation_lamb * TotalCorre_mss

## time step
time_step = tf.placeholder(tf.int32)

## training tensors 
tf_all_vars = tf.trainable_variables()
encodervar  = [var for var in tf_all_vars if var.name.startswith("EncoderX2Z")]
decodervar  = [var for var in tf_all_vars if var.name.startswith("DecoderZ2X")]

optimizer_vae = tf.train.AdamOptimizer(1e-4)
opt_vae = optimizer_vae.minimize(obj_vae, var_list = encodervar + decodervar)

saver = tf.train.Saver()
global_step = tf.Variable(0, name = 'global_step', trainable = False, dtype = tf.int32)

sess = tf.InteractiveSession()	
init = tf.global_variables_initializer().run()
assign_step_zero = tf.assign(global_step, 0)
init_step = sess.run(assign_step_zero)

#################################################################################
# training the networks
#################################################################################
x_input = data_matrix.copy()
index_shuffle = list(range(opt.num_cells_train))
current_step = 0

for epoch in range(opt.n_train_epochs):
	# shuffling the data per epoch
	np.random.shuffle(index_shuffle)
	x_input = x_input[index_shuffle, :]

	for i in range(0, opt.num_cells_train // opt.vae_batch_size):

		# train VAE/beta-TCVAE in each minibatch
		x_data = sample_X(x_input, opt.vae_batch_size)
		z_data = noise_prior(opt.vae_batch_size, opt.code_size)
		sess.run([opt_vae], {X_v : x_data, z_v: z_data, time_step : current_step})

		current_step += 1

	obj_vae_value, TC_value = sess.run([obj_vae, TotalCorre_mss], {X_v : x_data, z_v: z_data, time_step : current_step})

	print('epoch: {}; iteration: {}; beta-TCVAE loss:{}; Total Correlation:{}'.format(epoch, current_step, obj_vae_value, TC_value))

#################################################################################
# saving the trained model
#################################################################################
model_file_path = opt.model_path + "models_tcvae"
saving_model = saver.save(sess, model_file_path, global_step = current_step)


