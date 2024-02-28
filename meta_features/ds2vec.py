#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#### THIS CODE IS MOSTLY ASSEMBLED FROM THE DATASET2VEC REPOSITORY
#### PAPER: https://link.springer.com/article/10.1007/s10618-021-00737-9
#### FULL CODE: https://github.com/hadijomaa/dataset2vec

"""
@author: hsjomaa
"""

import argparse
import json
import os
import random

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

random.seed(3718)
tf.random.set_seed(0)
np.random.seed(42)

ARCHITECTURES = ['SQU','ASC','DES','SYM','ENC']

## taken from modules.py - 
def importance(config):
    if config['importance'] == 'linear':
        fn = lambda x:x
    elif config['importance'] == 'None':
        fn = None
    else:
        raise('please define n importance function')
    return fn
    
def get_units(idx,neurons,architecture,layers=None):
    assert architecture in ARCHITECTURES
    if architecture == 'SQU':
        return neurons
    elif architecture == 'ASC':
        return (2**idx)*neurons
    elif architecture == 'DES':
        return (2**(layers-1-idx))*neurons    
    elif architecture=='SYM':
        assert (layers is not None and layers > 2)
        if layers%2==1:
            return neurons*2**(int(layers/2) - abs(int(layers/2)-idx))
        else:
            x = int(layers/2)
            idx  = idx if idx < x else 2*x-idx -1
            return neurons*2**(int(layers/2) - abs(int(layers/2)-idx))
        
    elif architecture=='ENC':
        assert (layers is not None and layers > 2)
        if layers%2==0:
            x = int(layers/2)
            idx  = idx if idx < x else 2*x-idx -1            
            return neurons*2**(int(layers/2)-1 -idx)
        else:
            x = int(layers/2)
            idx  = idx if idx < x else 2*x-idx
            return neurons*2**(int(layers/2) -idx)

class Function(tf.keras.layers.Layer):

    def __init__(self, units, nhidden, nonlinearity,architecture,trainable):
        
        super(Function, self).__init__()
        
        self.n            = nhidden
        self.units        = units        
        self.nonlinearity = tf.keras.layers.Activation(nonlinearity)
        self.block        = [tf.keras.layers.Dense(units=get_units(_,self.units,architecture,self.n),trainable=trainable) \
                             for _ in range(self.n)]
            
    def call(self):
        raise Exception("Call not implemented")

class ResidualBlock(Function):

    def __init__(self, units, nhidden, nonlinearity,architecture,trainable):

        super().__init__(units, nhidden, nonlinearity,architecture,trainable)
            
    def call(self, x):
        e = x + 0
        for i, layer in enumerate(self.block):
            e = layer(e)
            if i < (self.n - 1):
                e = self.nonlinearity(e)
        return self.nonlinearity(e + x)
        
class FunctionF(Function):
    
    def __init__(self, units, nhidden, nonlinearity,architecture,trainable, resblocks=0):
        # m number of residual blocks
        super().__init__(units, nhidden, nonlinearity,architecture,trainable)
        # override function with residual blocks
        self.resblocks=resblocks
        if resblocks>0:
            self.block        = [tf.keras.layers.Dense(units=self.units,trainable=trainable)]
            assert(architecture=="SQU")
            self.block       += [ResidualBlock(units=self.units,architecture=architecture,nhidden=self.n,trainable=trainable,nonlinearity=self.nonlinearity) \
                                 for _ in range(resblocks)]                
            self.block       += [tf.keras.layers.Dense(units=self.units,trainable=trainable)]
        
    def call(self, x):
        e = x
        
        for i,fc in enumerate(self.block):
            
            e = fc(e)
            
            # make sure activation only applied once!
            if self.resblocks == 0:
                e = self.nonlinearity(e)
            else:
                # only first one
                if i==0 or i == (len(self.block)-1):
                    e = self.nonlinearity(e)    

        return e

class PoolF(tf.keras.layers.Layer):

    def __init__(self,units):    
        super(PoolF, self).__init__()
        
        self.units = units
        
    def call(self,x,nclasses,nfeature,ninstanc):
        
        s = tf.multiply(nclasses,tf.multiply(nfeature,ninstanc))
        
        x           = tf.split(x,num_or_size_splits=s,axis=0)
        
        e  = []
        
        for i,bx in enumerate(x):
            
            te     = tf.reshape(bx,shape=(1,nclasses[i],nfeature[i],ninstanc[i],self.units))
            
            te     = tf.reduce_mean(te,axis=3)
            e.append(tf.reshape(te,shape=(nclasses[i]*nfeature[i],self.units)))
            
        e = tf.concat(e,axis=0)
        
        return e
    
class FunctionG(Function):
    def __init__(self, units, nhidden, nonlinearity,architecture,trainable):
        
        super().__init__(units, nhidden, nonlinearity,architecture,trainable)
    def call(self, x):
        e = x
        
        for fc in self.block:
            
            e = fc(e)
            
            e = self.nonlinearity(e)

        return e

class PoolG(tf.keras.layers.Layer):

    def __init__(self,units):    
        super(PoolG, self).__init__()
        
        self.units = units
        
    def call(self, x,nclasses,nfeature):
        
        s = tf.multiply(nclasses, nfeature)      
        
        x = tf.split(x,num_or_size_splits=s,axis=0)
        
        e  = []
        
        for i,bx in enumerate(x):
            
            te     = tf.reshape(bx,shape=(1,nclasses[i]*nfeature[i],self.units))
            
            te     = tf.reduce_mean(te,axis=1)
            
            e.append(te)
            
        e = tf.concat(e,axis=0)

        return e
    
class FunctionH(Function):
    def __init__(self, units, nhidden, nonlinearity,architecture,trainable, resblocks=0):
        # m number of residual blocks
        super().__init__(units, nhidden, nonlinearity,architecture,trainable)
        # override function with residual blocks
        self.resblocks = resblocks
        if resblocks>0:
            self.block        = [tf.keras.layers.Dense(units=self.units,trainable=trainable)]
            assert(architecture=="SQU")
            self.block       += [ResidualBlock(units=self.units,architecture=architecture,nhidden=self.n,trainable=trainable,nonlinearity=self.nonlinearity) \
                                 for _ in range(resblocks)]                
            self.block       += [tf.keras.layers.Dense(units=self.units,trainable=trainable)]
        
    def call(self,x):
        
        e = x
        
        for i,fc in enumerate(self.block):
            
            e = fc(e)
            # make sure activation only applied once!
            if self.resblocks == 0:
                if i<(len(self.blocks)-1):
                    e = self.nonlinearity(e)
            else:
                # only first one
                if i==0:
                    e = self.nonlinearity(e)      

        return e

class PoolH(tf.keras.layers.Layer):
    def __init__(self, batch_size,units):
        """
        """
        super(PoolH, self).__init__()
        self.batch_size = batch_size
        self.units = units
        
    def call(self, x,ignore_negative):
        
        e  =  tf.reshape(x,shape=(self.batch_size,3,self.units))
        # average positive meta-features
        e1 = tf.reduce_mean(e[:,:2],axis=1)
        if not ignore_negative:
            # select negative meta-feautures 
            e1 = e[:,-1][:,None]            
        # reshape, i.e. output is [batch_size,nhidden]
        e  = tf.reshape(e1,shape=(self.batch_size,self.units))            
        
        return e


## taken from sampling.py
class Batch(object):
    
    def __init__(self,batch_size,fixed_shape = True):
        
        self.batch_size = batch_size
        self.fixed_shape = fixed_shape
        self.clear()
    
    def clear(self):
        # flattened triplets
        self.x = []
        # number of instances per item in triplets
        self.instances = []
        # number of features per item in triplets
        self.features = []
        # number of classes per item in triplets
        self.classes = []
        # model input
        self.input = None
        
    def append(self,instance):
        
        if len(self.x)==self.batch_size:
            
            self.clear()
            
        self.x.append(instance[0])
        self.instances.append(instance[1])
        self.features.append(instance[2])
        self.classes.append(instance[3])
        
    def collect(self):
        
        if len(self.x)!= self.batch_size and self.fixed_shape:
            raise(f'Batch formation incomplete!\n{len(self.x)}!={self.batch_size}')
        self.input = (tf.concat(self.x,axis=0),
                      tf.cast(tf.transpose(tf.concat(self.classes,axis=0)),dtype=tf.int32),
                      tf.cast(tf.transpose(tf.concat(self.features,axis=0)),dtype=tf.int32),
                      tf.cast(tf.transpose(tf.concat(self.instances,axis=0)),dtype=tf.int32),
                      )
        self.output = {'similaritytarget':tf.concat([tf.ones(self.batch_size),tf.zeros(self.batch_size)],axis=0)}

def pool(n,ntotal,shuffle):
    _pool = [_ for _ in list(range(ntotal)) if _!= n]
    if shuffle:
        random.shuffle(_pool)
    return _pool

class Sampling(object):
    def __init__(self,dataset):
        self.dataset          = dataset
        self.distribution     = pd.DataFrame(data=None,columns=['targetdataset','sourcedataset'])
        self.targetdataset   = None

    def sample(self,batch,split,sourcesplit):
        
        nsource  = len(self.dataset.orig_data[sourcesplit])
        ntarget  = len(self.dataset.orig_data[split])
        targetdataset = np.random.choice(ntarget,batch.batch_size)
        # clear batch
        batch.clear() 
        # find the negative dataset list of batch_size
        sourcedataset = []
        for target in targetdataset:
            if split==sourcesplit:
                swimmingpool  = pool(target,nsource,shuffle=True)  
            else:
                swimmingpool  = pool(-1,nsource,shuffle=True)
            sourcedataset.append(np.random.choice(swimmingpool))
        sourcedataset = np.asarray(sourcedataset).reshape(-1,)
        for target,source in zip(targetdataset,sourcedataset):
            # build instance
            instance = self.dataset.instances(target,source,split=split,sourcesplit=sourcesplit)
            batch.append(instance)
        
        distribution      = np.concatenate([targetdataset.reshape(-1,1),sourcedataset[:,None]],axis=1)
        self.distribution = pd.concat([self.distribution,\
                                       pd.DataFrame(distribution,columns=['targetdataset','sourcedataset'])],axis=0,ignore_index=True)
            
        self.targetdataset   = targetdataset  
        return batch
    
class TestSampling(object):
    def __init__(self,dataset):
        self.dataset          = dataset
        self.distribution     = pd.DataFrame(data=None,columns=['targetdataset','sourcedataset'])

    def sample(self,batch,split,sourcesplit,targetdataset):
        
        nsource  = len(self.dataset.orig_data[sourcesplit])
        # clear batch
        batch.clear() 
        # find the negative dataset list of batch_size
        swimmingpool  = pool(targetdataset,nsource,shuffle=True) if split==sourcesplit else pool(-1,nsource,shuffle=True)
        # double check divisibilty by batch size
        sourcedataset = np.random.choice(swimmingpool,batch.batch_size,replace=False)
        # iterate over batch negative datasets
        for source in sourcedataset:
            # build instance
            instance = self.dataset.instances(targetdataset,source,split=split,sourcesplit=sourcesplit)
            batch.append(instance)
        
        distribution      = np.concatenate([np.asarray(batch.batch_size*[targetdataset])[:,None],sourcedataset[:,None]],axis=1)
        self.distribution = pd.concat([self.distribution,\
                                       pd.DataFrame(distribution,columns=['targetdataset','sourcedataset'])],axis=0,ignore_index=True)    
            
        return batch

    def sample_from_one_dataset(self,batch):
        
        # clear batch
        batch.clear() 
        # iterate over batch negative datasets
        for _ in range(batch.batch_size):
            # build instance
            instance = self.dataset.instances()
            batch.append(instance)
        
        return batch


## taken from dummdataset.py
def ptp(X):
    param_nos_to_extract = X.shape[1]
    domain = np.zeros((param_nos_to_extract, 2))
    for i in range(param_nos_to_extract):
        domain[i, 0] = np.min(X[:, i])
        domain[i, 1] = np.max(X[:, i])
    X = (X - domain[:, 0]) / np.ptp(domain, axis=1)
    return X

def flatten(x,y):
    '''
    Genearte x_i,y_j for all i,j \in |x|

    Parameters
    ----------
    x : numpy.array
        predictors; shape = (n,m)
    y : numpy.array
        targets; shape = (n,t)

    Returns
    -------
    numpy.array()
        shape ((n\times m\times t)\times 2).

    '''
    x_stack = []
    for c in range(y.shape[1]):
        c_label = np.tile(y[:,c],reps=[x.shape[1]]).transpose().reshape(-1,1)
        x_stack.append(np.concatenate([x.transpose().reshape(-1,1),c_label],axis=-1))
    return np.vstack(x_stack)

class Dataset(object):
    
    def __init__(self,file,rootdir):
            # read dataset
            self.X,self.y = self.__get_data(file,rootdir=rootdir)

            # batch properties
            self.ninstanc = 256
            self.nclasses = 3
            self.nfeature = 16
            
             
    def __get_data(self,file,rootdir):

        # read dataset folds
        datadir = os.path.join(rootdir, "datasets", file)
        # read internal predictors
        data = pd.read_csv(f"{datadir}/{file}_py.dat",header=None)
        # transform to numpy
        data    = np.asarray(data)
        # read internal target
        labels = pd.read_csv(f"{datadir}/labels_py.dat",header=None)
        # transform to numpy
        labels    = np.asarray(labels)        

        return data,labels

    def sample_batch(self,data,labels,ninstanc,nclasses,nfeature):
        '''
        Sample a batch from the dataset of size (ninstanc,nfeature)
        and a corresponding label of shape (ninstanc,nclasses).

        Parameters
        ----------
        data : numpy.array
            dataset; shape (N,F) with N >= nisntanc and F >= nfeature
        labels : numpy.array
            categorical labels; shape (N,) with N >= nisntanc
        ninstanc : int
            Number of instances in the output batch.
        nclasses : int
            Number of classes in the output label.
        nfeature : int
            Number of features in the output batch.

        Returns
        -------
        data : numpy.array
            subset of the original dataset.
        labels : numpy.array
            one-hot encoded label representation of the classes in the subset

        '''
        # Create the one-hot encoder
        ohc           = OneHotEncoder(categories = [range(len(np.unique(labels)))],sparse_output=False)
        d = {ni: indi for indi, ni in enumerate(np.unique(labels))}
        # process the labels
        labels        = np.asarray([d[ni] for ni in labels.reshape(-1)]).reshape(-1)
        # transform the labels to one-hot encoding
        labels        = ohc.fit_transform(labels.reshape(-1,1))
        # ninstance should be less than or equal to the dataset size
        ninstanc            = np.random.choice(np.arange(0,data.shape[0]),size=np.minimum(ninstanc,data.shape[0]),replace=False)
        # nfeature should be less than or equal to the dataset size
        nfeature         = np.random.choice(np.arange(0,data.shape[1]),size=np.minimum(nfeature,data.shape[1]),replace=False)
        # nclasses should be less than or equal to the total number of labels
        nclasses         = np.random.choice(np.arange(0,labels.shape[1]),size=np.minimum(nclasses,labels.shape[1]),replace=False)
        # extract data at selected instances
        data          = data[ninstanc]
        # extract labels at selected instances
        labels        = labels[ninstanc]
        # extract selected features from the data
        data          = data[:,nfeature]
        # extract selected labels from the data
        labels        = labels[:,nclasses]
        return data,labels
    
    def instances(self,ninstanc=None,nclasses=None,nfeature=None):
        # check if ninstance is provided
        ninstanc = ninstanc if ninstanc is not None else self.ninstanc
        # check if ninstance is provided
        nclasses = nclasses if nclasses is not None else self.nclasses
        # check if ninstance is provided
        nfeature = nfeature if nfeature is not None else self.nfeature        
        # check if neg batch is provided
        instance_x,instance_i = [],[]
        # append information to the placeholders
        x,y = self.sample_batch(self.X,self.y,ninstanc,nclasses,nfeature)
        instance_i.append(x.shape+(y.shape[1],)+(-1,))
        instance_x.append(flatten(x,y))
        # remove x,y
        del x,y
        # stack x values
        x = np.vstack(instance_x)
        # stack ninstanc
        ninstance = np.vstack(instance_i)[:,0][:,None]
        # stack nfeatures
        nfeature = np.vstack(instance_i)[:,1][:,None]
        # stack nclasses
        nclasses = np.vstack(instance_i)[:,2][:,None]
        # get task description of surr task
        return x,ninstance,nfeature,nclasses


## taken from extract_meta_features.py
def Dataset2VecModel(configuration):

    nonlinearity_d2v  = configuration['nonlinearity_d2v']
    # Function F
    units_f     = configuration['units_f']
    nhidden_f   = configuration['nhidden_f']
    architecture_f = configuration['architecture_f']
    resblocks_f = configuration['resblocks_f']

    # Function G
    units_g     = configuration['units_g']
    nhidden_g   = configuration['nhidden_g']
    architecture_g = configuration['architecture_g']
    
    # Function H
    units_h     = configuration['units_h']
    nhidden_h   = configuration['nhidden_h']
    architecture_h = configuration['architecture_h']
    resblocks_h   = configuration['resblocks_h']
    #
    batch_size = configuration["batch_size"]
    trainable = False
    # input two dataset2vec shape = [None,2], i.e. flattened tabular batch
    x      = tf.keras.Input(shape=(2),dtype=tf.float32)
    # Number of sampled classes from triplets
    nclasses = tf.keras.Input(shape=(batch_size),dtype=tf.int32,batch_size=1)
    # Number of sampled features from triplets
    nfeature = tf.keras.Input(shape=(batch_size),dtype=tf.int32,batch_size=1)
    # Number of sampled instances from triplets
    ninstanc = tf.keras.Input(shape=(batch_size),dtype=tf.int32,batch_size=1)
    # Encode the predictor target relationship across all instances
    layer    = FunctionF(units = units_f,nhidden = nhidden_f,nonlinearity = nonlinearity_d2v,architecture=architecture_f,resblocks=resblocks_f,trainable=trainable)(x)
    # Average over instances
    layer    = PoolF(units=units_f)(layer,nclasses[0],nfeature[0],ninstanc[0])
    # Encode the interaction between features and classes across the latent space
    layer    = FunctionG(units = units_g,nhidden   = nhidden_g,nonlinearity = nonlinearity_d2v,architecture = architecture_g,trainable=trainable)(layer)
    # Average across all instances
    layer    = PoolG(units=units_g)(layer,nclasses[0],nfeature[0])
    # Extract the metafeatures
    metafeatures    = FunctionH(units = units_h,nhidden   = nhidden_h, nonlinearity = nonlinearity_d2v,architecture=architecture_h,trainable=trainable,resblocks=resblocks_h)(layer)
    # define hierarchical dataset representation model
    dataset2vec     = tf.keras.Model(inputs=[x,nclasses,nfeature,ninstanc], outputs=metafeatures)
    return dataset2vec


## custom code

class CustomDataset(Dataset):

    def __init__(self, data, targets):
        self.ninstanc = 256
        self.nclasses = 3
        self.nfeature = 16
        self.X,self.y = data, targets # simple store the arrays instead of loading them with __get_data
                      
def extract(data, targets, n_samples=10, d2v_model_root='ds2vec_models'):
    dataset = CustomDataset(data, targets)
    testsampler = TestSampling(dataset=dataset)
    metafeatures = []
    d2v_model_root = os.path.join(os.path.dirname(__file__), d2v_model_root)
    try:
        for subdir in os.listdir(d2v_model_root):
            # load model from this split
            log_dir = os.path.join(d2v_model_root, subdir)
            configuration = json.load(open(os.path.join(log_dir, "configuration.txt"),"r"))
            batch = Batch(configuration['batch_size'])
            model = Dataset2VecModel(configuration)
            model.load_weights(os.path.join(log_dir, "weights"), by_name=False, skip_mismatch=False)
            # calculate features by sampling from the dataset
            datasetmf = []
            for _ in range(n_samples): # any number of samples
                batch = testsampler.sample_from_one_dataset(batch)
                batch.collect()
                datasetmf.append(model(batch.input).numpy())
            metafeatures.append(np.vstack(datasetmf).mean(axis=0))
        # average the features of all CV split models
        metafeatures = np.array(metafeatures).mean(axis=0)
    except:
        metafeatures = np.full((32,), fill_value=np.nan)
    return pd.Series(metafeatures, index=[f'ft{idx}' for idx in range(metafeatures.size)])
