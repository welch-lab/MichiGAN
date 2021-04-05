# MichiGAN: Learning disentangled representations of single-cell data for high-quality generation

## Sampling from disentangled representations of single-cell data using generative adversarial networks

The current folder contains files for implementing **PCA, GMM, VAE/beta-TCVAE, WGAN-GP, InfoWGAN-GP, ssInfoWGAN-GP, CWGAN-GP and MichiGAN** on single-cell RNA-seq data. See our preprint for details:  
[Sampling from disentangled representations of single-cell data using generative adversarial networks](https://www.biorxiv.org/content/10.1101/2021.01.15.426872v1) (Yu and Welch, 2021+). We have a [presentation video](https://youtu.be/5tsccPMPzLQ) for [Learning Meaningful Representations of Life Workshop](https://www.lmrl.org/) at NeurIPS 2020, where we named our framework as `DRGAN` and changed the name to `MichiGAN` afterwards.   

## List of Files:

1) `/data` is the folder containing the real scRNA-seq dataset of Tabula Muris heart data. Users can download the SCANPY-processed data on https://www.dropbox.com/sh/xseb0u6p01te3vr/AACuskVfswUFn5MroEFrqI-Xa?dl=0. 
2) `/FunctionProgramExamples/examples` is the folder for the experiments of\
  (1) `vae.py`: VAE; \
  (2) `beta_tcvae.py`: beta-TCVAE;\
  (3) `wgangp.py`: WGAN-GP;\
  (4) `infowgangp.py`: InfoWGAN-GP;\
  (5) `MichiGAN_mean.py`: MichiGAN on mean representations;\
  (6) `MichiGAN_sample.py`: MichiGAN on sampled representations\
  on the Tabula Muris heart dataset.
  The ipython notebooks `example_**.ipynb` also give examples of how to train different deep generative models on Tabula Muris heart data. 

3) `/FunctionProgramExamples/Adam_prediction.py` is the StableGAN implementation file on https://github.com/taki0112/StableGAN-Tensorflow for the GAN-based methods
4) `/FunctionProgramExamples/lib.py` contains the Python and TensorFlow functions
3) `/FunctionProgramExamples/nets.py` has the network architectures for different deep generative models 

## Notes:  

1) The example program demonstrates the training for Tabula Muris data with 4221 cells and 4062 genes processed by the SCANPY package and stored as .npy file. 
2) The related module versions are: 
```
 (1) Python 3.6
 (2) numpy: 1.16.3
 (3) pandas 0.25.3 
 (4) scanpy: 1.4.6
 (5) tensorflow: 1.14.0 
```
