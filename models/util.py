#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np 
import sys 
import pandas as pd 
from random import shuffle
from scipy import sparse


def tf_log(value):
    """
    tensorflow logrithmic 
    """
    return tf.log(value + 1e-16)
    
def logsumexp(value,  dim = None, keepdims = False):
    """
    calculate q(Z) = sum_{X}[sum_{j}q(Zj|X)] and q(Zj) = sum_{X}[q(Zj|X)]
    """
    if dim is not None:
        m = tf.reduce_max(value, axis = dim, keepdims = True)
        value0 = tf.subtract(value, m)
        if keepdims is False:
            m = tf.squeeze(m, dim)
        return tf.add(m, tf_log(tf.reduce_sum(tf.exp(value0), axis = dim, keepdims = keepdims)))

    else:
        m = tf.reduce_max(value)
        sum_exp = tf.reduce_sum(tf.exp(tf.subtract(value, m)))	
        return tf.add(m, tf_log(sum_exp))


def total_correlation(marginal_entropies, joint_entropies):
    """
    calculate total correlation from the marginal and joint entropies
    """
    return tf.reduce_sum(marginal_entropies) - tf.reduce_sum(joint_entropies)


def shuffle_adata(adata):
    """
    shuffle adata
    """
    if sparse.issparse(adata.X):
        adata.X = adata.X.A
    
    ind_list = list(range(adata.shape[0]))
    shuffle(ind_list)
    new_adata = adata[ind_list, :]
    
    return new_adata


def shuffle_adata_cond(adata, cond):
    """
    Shuffle adata with the label 
    """
    
    if sparse.issparse(adata.X):
        adata.X = adata.X.A
    ind_list = list(range(adata.shape[0]))
    shuffle(ind_list)

    new_adata = adata[ind_list, :]
    new_cond = cond[ind_list, :]
    
    return new_adata, new_cond