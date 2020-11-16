# MichiGAN: Learning disentangled representations of single-cell data for high-quality generation

## Predicting unobserved cell states from disentangled representations of single-cell data using generative adversarial networks

The current folder contains files for implementing **VAE/beta-TCVAE, WGAN-GP, InfoWGAN-GP, MichiGAN** on single-cell RNA-seq data described in 
**"Predicting unobserved cell states from disentangled representations of single-cell data using generative adversarial networks"** by Yu and Welch (2020+).

## List of Files:

1) `/data` is te folder containing the real scRNA-seq dataset of Tabula Muris heart data. 
2) `/examples` is the folder for the experiments of 

  (1) `vae.py`: VAE
  (2) `beta_tcvae.py`: beta-TCVAE
  (3) `wgangp.py`: WGAN-GP
  (4) `infowgangp.py`: InfoWGAN-GP
  (5) `MichiGAN_mean.py`: MichiGAN on mean representations
  (6) `MichiGAN_sample.py`: MichiGAN on sampled representations
  on the Tabula Muris heart dataset. The ipython notebooks also give the examples of training different deep generative models on Tabula Muris heart data. 

3) Adam_prediction.py is the StableGAN implementation file on https://github.com/taki0112/StableGAN-Tensorflow for the GAN-based methods
4) lib.py contains the Python and TensorFlow functions
3) nets.py has the network architectures for different deep generative models 

NOTES:  1) The example program demonstrates the training for Tabula Muris data with 4221 cells and 4062 genes processed by the SCANPY package and stored as .npy file. 
	   2) The modules version is: Python 3.6
		 (1) numpy: 1.16.3; (2) pandas 0.25.3 (3) 1.4.6
		 (4) TensorFlow: 1.14.0 
