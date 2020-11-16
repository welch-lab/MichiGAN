"""
Code for training WGAN-GP on Tabula Muris heart data
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
		self.n_train_epochs = 10                 # number of epochs for training
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
		self.model_path = "./examples/models_wgangp/"        # path saving the model


#################################################################################
# load data and define hyperparameters
#################################################################################
data_matrix = np.load('./data/TabulaMurisHeart_Processed.npy')
data_meta = pd.read_csv("./data/TabulaMurisHeart_MetaInformation.csv")
opt = Options(data_matrix.shape[0], data_matrix.shape[1])

#################################################################################
# define network tensors
#################################################################################
z = tf.placeholder(tf.float32, shape = (None, opt.code_size))
X = tf.placeholder(tf.float32, shape = (None, opt.gex_size))

## generator
X_gen_data = gan_generator(z, opt)


## discriminator 
Dx_real, Dx_hidden_real = gan_discriminator(X, opt)
Dx_fake, Dx_hidden_fake = gan_discriminator(X_gen_data, opt, reuse = True)

## discriminator loss
obj_d_or = tf.reduce_mean(Dx_real) - tf.reduce_mean(Dx_fake) 
gradient_penalty = compute_gp(X, X_gen_data, gan_discriminator, opt)
obj_d = obj_d_or + opt.GradientPenaly_lambda * gradient_penalty

## generator loss
obj_g = tf.reduce_mean(Dx_fake) 

## time step
time_step = tf.placeholder(tf.int32)

## training tensors 
tf_all_vars = tf.trainable_variables()
dvar  = [var for var in tf_all_vars if var.name.startswith("Discriminator")]
gvar  = [var for var in tf_all_vars if var.name.startswith("Generator")]


### Thanks to taki0112 for the TF StableGAN implementation https://github.com/taki0112/StableGAN-Tensorflow
from Adam_prediction import Adam_Prediction_Optimizer
opt_g = Adam_Prediction_Optimizer(learning_rate = 1e-4, beta1 = 0.9, beta2 = 0.999, prediction = True).minimize(obj_g, var_list = gvar)
opt_d = Adam_Prediction_Optimizer(learning_rate = 1e-4, beta1 = 0.9, beta2 = 0.999, prediction = False).minimize(obj_d, var_list = dvar)


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
	
	for i in range(0, opt.num_cells_train // opt.batch_size):
		# train discriminator opt.Diters times per training of generator
		for k in range(opt.Diters):

			x_data = sample_X(x_input, opt.batch_size)
			z_data = noise_prior(opt.batch_size, opt.code_size)

			sess.run([opt_d], {X : x_data, z: z_data, time_step : current_step})


		# train generator
		x_data = sample_X(x_input, opt.batch_size)
		z_data = noise_prior(opt.batch_size, opt.code_size)
		sess.run([opt_g], {X : x_data, z: z_data, time_step : current_step})

		current_step += 1
	
	obj_d_value, obj_g_value = sess.run([obj_d, obj_g], {X : x_data, z: z_data, time_step : current_step})
	
	print('epoch: {}; iteration: {}; generator loss: {}; discriminator loss:{}'.format(epoch, current_step, obj_g_value, obj_d_value))

#################################################################################
# saving the trained model
#################################################################################
model_file_path = opt.model_path + "models_wgangp"
saving_model = saver.save(sess, model_file_path, global_step = current_step)
