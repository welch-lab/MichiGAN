"""
neural networks
"""
import tensorflow as tf
from tensorflow import distributions as ds

# VAE/beta-TCVAE networks
def encoder1(x, opt, reuse = False):
	""" encoder network """
	with tf.compat.v1.variable_scope("encoder1", reuse = reuse):

		en_dense1 = tf.layers.dense(inputs = x, 
			units = opt.inflate_to_size2, 
			activation = None, 
			name = "encoder_dense1", 
			kernel_initializer = tf.contrib.layers.xavier_initializer())
		en_dense1 = tf.layers.batch_normalization(en_dense1)
		en_dense1 = tf.nn.leaky_relu(en_dense1)
		en_dense1 = tf.layers.dropout(en_dense1, opt.dropout_rate)

		en_dense2 = tf.layers.dense(inputs = en_dense1, 
			units = opt.inflate_to_size1,
			activation = None, 
			name = "encoder_dense2", 
			kernel_initializer = tf.contrib.layers.xavier_initializer())
		en_dense2 = tf.layers.batch_normalization(en_dense2)
		en_dense2 = tf.nn.relu(en_dense2)
		en_dense2 = tf.layers.dropout(en_dense2, opt.dropout_rate)

		en_loc = tf.layers.dense(inputs = en_dense2, 
			units = opt.code_size,
			activation = None, 
			name = "encoder_loc")

		en_scale = tf.layers.dense(inputs = en_dense2, 
			units = opt.code_size,
			activation = None, 
			name = "encoder_scale")
		en_scale = tf.nn.softplus(en_scale)

		return en_loc, en_scale 



def decoder2(z, opt, reuse = False):
	""" decoder network """
	with tf.compat.v1.variable_scope("decoder2", reuse=reuse):

		de_dense1 = tf.layers.dense(inputs = z,
			units = opt.inflate_to_size1,
			activation = None,
			name = "decoder_dense1",
			kernel_initializer = tf.contrib.layers.xavier_initializer())
		de_dense1 = tf.layers.batch_normalization(de_dense1)
		de_dense1 = tf.nn.leaky_relu(de_dense1)
		de_dense1 = tf.layers.dropout(de_dense1, opt.dropout_rate)
		

		de_dense2 = tf.layers.dense(inputs = de_dense1,
			units = opt.inflate_to_size2,
			activation = None,
			name = "decoder_dense2",
			kernel_initializer = tf.contrib.layers.xavier_initializer())
		de_dense2 = tf.layers.batch_normalization(de_dense2)
		de_dense2 = tf.nn.leaky_relu(de_dense2)
		de_dense2 = tf.layers.dropout(de_dense2, opt.dropout_rate)

		de_loc = tf.layers.dense(inputs = de_dense2,
			units = opt.gex_size,
			activation = None,
			name = "decoder_loc")

		de_scale = tf.ones_like(de_loc)


		return ds.Normal(de_loc, de_scale)


# GAN networks
def generator(z, opt, reuse = False):
	""" generator network """
	with tf.compat.v1.variable_scope("generator", reuse = reuse):

		de_dense1 = tf.layers.dense(inputs = z,
			units = opt.inflate_to_size1,
			activation = None,
			name = "generator_dense1",
			kernel_initializer = tf.contrib.layers.xavier_initializer())
		de_dense1 = tf.layers.batch_normalization(de_dense1)
		de_dense1 = tf.nn.leaky_relu(de_dense1)
		de_dense1 = tf.layers.dropout(de_dense1, opt.dropout_rate)

		de_dense2 = tf.layers.dense(inputs = de_dense1,
			units = opt.inflate_to_size2,
			activation = None,
			name = "generator_dense2",
			kernel_initializer = tf.contrib.layers.xavier_initializer())
		de_dense2 = tf.layers.batch_normalization(de_dense2)
		de_dense2 = tf.nn.leaky_relu(de_dense2)
		de_dense2 = tf.layers.dropout(de_dense2, opt.dropout_rate)

		de_dense3 = tf.layers.dense(inputs = de_dense2,
			units = opt.inflate_to_size3,
			activation = None,
			name = "generator_dense3",
			kernel_initializer = tf.contrib.layers.xavier_initializer())
		de_dense3 = tf.layers.batch_normalization(de_dense3)
		de_dense3 = tf.nn.relu(de_dense3)
		de_dense3 = tf.layers.dropout(de_dense3, opt.dropout_rate)

		de_output = tf.layers.dense(inputs = de_dense3,
			units = opt.gex_size,
			activation = None,
			name = "generator_output")

		return de_output

def discriminator(x, opt, reuse = False):
	""" discriminator network """
	with tf.compat.v1.variable_scope("discriminator", reuse = reuse):

		disc_dense1 = tf.layers.dense(inputs = x,
			units = opt.disc_internal_size1,
			activation = None,
			name = "disc_dense1",
			kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 0.8),
			kernel_initializer = tf.contrib.layers.xavier_initializer())
		disc_dense1 = tf.layers.batch_normalization(disc_dense1)
		disc_dense1 = tf.nn.leaky_relu(disc_dense1)

		disc_dense2 = tf.layers.dense(inputs = disc_dense1,
			units = opt.disc_internal_size2,
			activation = None,
			name = "disc_dense2",
			kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 0.8),
			kernel_initializer = tf.contrib.layers.xavier_initializer())
		disc_dense2 = tf.layers.batch_normalization(disc_dense2)
		disc_dense2 = tf.nn.leaky_relu(disc_dense2)

		disc_dense3 = tf.layers.dense(inputs = disc_dense2,
			units= opt.disc_internal_size3,
			activation = None,
			name = "disc_dense3",
			kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 0.8),
			kernel_initializer = tf.contrib.layers.xavier_initializer())
		disc_dense3 = tf.layers.batch_normalization(disc_dense3)
		disc_dense3 = tf.nn.relu(disc_dense3)

		disc_output = tf.layers.dense(inputs = disc_dense3,
			units = 1,
			activation = None,
			name = "disc_output")
		
		return disc_output, disc_dense3


def mutual_discriminator(x, opt, reuse = False):
	""" InfoGAN discriminator """
	with tf.compat.v1.variable_scope("infogan_discriminator", reuse=reuse):

		disc_dense1 = tf.layers.dense(inputs= x,
			units = opt.disc_internal_size1,
			activation = None,
			name = "disc_dense1",
			kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 0.8),
			kernel_initializer = tf.contrib.layers.xavier_initializer())
		disc_dense1 = tf.layers.batch_normalization(disc_dense1)
		disc_dense1 = tf.nn.leaky_relu(disc_dense1)

		disc_dense2 = tf.layers.dense(inputs=disc_dense1,
			units = opt.disc_internal_size2,
			activation = None,
			name = "disc_dense2",
			kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 0.8),
			kernel_initializer = tf.contrib.layers.xavier_initializer())
		disc_dense2 = tf.layers.batch_normalization(disc_dense2)
		disc_dense2 = tf.nn.leaky_relu(disc_dense2)

		disc_dense3 = tf.layers.dense(inputs = disc_dense2,
			units = opt.disc_internal_size3,
			activation = None,
			name = "disc_dense3",
			kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 0.8),
			kernel_initializer = tf.contrib.layers.xavier_initializer())
		disc_dense3 = tf.layers.batch_normalization(disc_dense3)
		disc_dense3 = tf.nn.relu(disc_dense3)

		disc_output = tf.layers.dense(inputs = disc_dense3,
			units = 1,
			activation = None,
			name = "disc_output")
		
		q_dense1 = tf.layers.dense(inputs = disc_dense3,
			units = opt.disc_internal_size3,
			activation = None,
			name = "mutual_dense",
			kernel_initializer = tf.contrib.layers.xavier_initializer())
		q_dense1 = tf.layers.batch_normalization(q_dense1)
		q_dense1 = tf.nn.leaky_relu(q_dense1)

		q_output = tf.layers.dense(inputs = q_dense1,
			units = (opt.code_size if opt.InfoGAN_fix_std else opt.code_size * 2), 
			activation = None,
			name = "mutual_output")


		return disc_output, q_output

# MichiGAN networks-Conditional GANs with projection discriminator
def con_generator(z, y, opt, reuse = False):
	""" conditional generator network """
	with tf.compat.v1.variable_scope("con_generator", reuse = reuse):
		zy = tf.concat([z, y], axis = 1)
		de_dense1 = tf.layers.dense(inputs = zy,
			units = opt.inflate_to_size1,
			activation = None,
			name = "congen_dense1",
			kernel_initializer = tf.contrib.layers.xavier_initializer())
		de_dense1 = tf.layers.batch_normalization(de_dense1)
		de_dense1 = tf.nn.leaky_relu(de_dense1)
		de_dense1 = tf.layers.dropout(de_dense1, opt.dropout_rate)

		de_dense2 = tf.layers.dense(inputs = de_dense1,
			units = opt.inflate_to_size2,
			activation = None,
			name = "congen_dense2",
			kernel_initializer = tf.contrib.layers.xavier_initializer())
		de_dense2 = tf.layers.batch_normalization(de_dense2)
		de_dense2 = tf.nn.leaky_relu(de_dense2)
		de_dense2 = tf.layers.dropout(de_dense2, opt.dropout_rate)

		de_dense3 = tf.layers.dense(inputs = de_dense2,
			units = opt.inflate_to_size3,
			activation = None,
			name = "congen_dense3",
			kernel_initializer = tf.contrib.layers.xavier_initializer())
		de_dense3 = tf.layers.batch_normalization(de_dense3)
		de_dense3 = tf.nn.relu(de_dense3)
		de_dense3 = tf.layers.dropout(de_dense3, opt.dropout_rate)

		de_output = tf.layers.dense(inputs = de_dense3,
			units = opt.gex_size,
			activation = None,
			name = "congen_output")

		return de_output


def con_discriminator(x, y, opt, reuse = False):
	""" conditional discriminator network """
	with tf.compat.v1.variable_scope("con_discriminator", reuse = reuse):

		disc_dense1 = tf.layers.dense(inputs = x,
			units = opt.disc_internal_size1,
			activation = None,
			name = "disc_dense1",
			kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 0.8),
			kernel_initializer = tf.contrib.layers.xavier_initializer())
		disc_dense1 = tf.layers.batch_normalization(disc_dense1)
		disc_dense1 = tf.nn.leaky_relu(disc_dense1)

		disc_dense2 = tf.layers.dense(inputs = disc_dense1,
			units = opt.disc_internal_size2,
			activation = None,
			name = "disc_dense2",
			kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 0.8),
			kernel_initializer = tf.contrib.layers.xavier_initializer())
		disc_dense2 = tf.layers.batch_normalization(disc_dense2)
		disc_dense2 = tf.nn.leaky_relu(disc_dense2)

		disc_dense3 = tf.layers.dense(inputs = disc_dense2,
			units = opt.disc_internal_size3,
			activation = None,
			name = "disc_dense3",
			kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 0.8),
			kernel_initializer = tf.contrib.layers.xavier_initializer())
		disc_dense3 = tf.layers.batch_normalization(disc_dense3)
		disc_dense3 = tf.nn.relu(disc_dense3)

		disc_output = tf.layers.dense(inputs = disc_dense3,
			units = 1,
			activation = None,
			name = "disc_output")
		
		disc_output1 = disc_output + tf.reduce_sum(y * disc_dense3, axis = 1, keepdims = True)
		
		return disc_output1, disc_dense3


# wrapping up networks
def vaes_encoder(x, opt, reuse = None):
	with tf.compat.v1.variable_scope("EncoderX2Z", reuse = reuse):
		mu, scale = encoder1(x, opt)
	return mu, scale

def vaes_decoder(z, opt, reuse = None):
	with tf.compat.v1.variable_scope("DecoderZ2X", reuse = reuse):
		h = decoder2(z, opt)
	return h

def gan_generator(z, opt, reuse = None):
	with tf.compat.v1.variable_scope("Generator", reuse = reuse):
		h = generator(z, opt)
	return h

def gan_discriminator(x, opt, reuse = None):
	with tf.compat.v1.variable_scope('Discriminator', reuse = reuse):
		f, f1 = discriminator(x, opt)
	return f, f1

def infogan_discriminator(x, opt, reuse = None):
	with tf.compat.v1.variable_scope('InfoGANDiscriminator', reuse = reuse):
		f, q_out = mutual_discriminator(x, opt)
	return f, q_out

def michigan_generator(z, y, opt, reuse = None):
	with tf.compat.v1.variable_scope("MichiGANGenerator", reuse=reuse):
		h = con_generator(z, y, opt)
	return h

def michigan_discriminator(x, y, opt, reuse = None):
	with tf.compat.v1.variable_scope('MichiGANDiscriminator', reuse = reuse):
		f, f1 = con_discriminator(x, y, opt)
	return f, f1

