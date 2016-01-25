# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 11:54:09 2016

@author: hedi

Uses file yielded by create_similarity_data.py for training a model that 
computes a projection of images and texts based upon our model.
"""
import numpy as np
import sys
from time import time
from sklearn.metrics.pairwise import cosine_similarity
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sim_model import Multimodal
import theano
fwrite = sys.stdout.write
dtype = theano.config.floatX

        
if __name__=='__main__':
    np.random.seed(1234)
    which_set = 'r'
    fwrite('Loading data...')
    data = np.load(which_set+'_similarity_data_save').all()
    fwrite(' Ok\n')
    pooling = 'mean' #mean, max or softmax
    optim = 'sgd' #sgd or rmsprop (but rmsprop is really slow...)
    folder = os.path.join(which_set+'_sim', pooling)
    if not os.path.exists(folder):
        os.mkdir(folder)
    saving_path = os.path.join(folder, "model.mod")
    
    d = data[data.keys()[0]]
    dim_img = d['image_emb'].shape[0]
    dim_txt = d['text_emb'].shape[1]
    dim_multi = 150
    
    image_embeddings = []
    text_embeddings = []
    product_ids = []
    fwrite('Loading data ...\n')
    product_ids = data.keys()
    for idx in product_ids:
        d = data.pop(idx)
        if not len(data)%5000:
            fwrite('%d\n' % len(data))
        text_embeddings.append(d['text_emb'].astype(dtype))
        image_embeddings.append(d['image_emb'].astype(dtype))
    del data
    n_data = len(product_ids)
    fwrite('Done\n')
    product_ids = np.array(product_ids[:n_data])
    image_embeddings = np.array(image_embeddings[:n_data])
    text_embeddings = np.array([t for t in text_embeddings[:n_data]])
    
    
    indexes = np.arange(n_data)
    np.random.shuffle(indexes)
    
    image_embeddings = image_embeddings[indexes]
    text_embeddings = text_embeddings[indexes]
    product_ids = product_ids[indexes]
    
    
    n_train = int(0.7*n_data)
    Xim_train = image_embeddings[:n_train]
    Xim_test = image_embeddings[n_train+1:]
    Xtxt_train = text_embeddings[:n_train]
    Xtxt_test = text_embeddings[n_train+1:]
    product_ids_train = product_ids[:n_train]
    product_ids_test = product_ids[n_train + 1 :]
    
    Xtxt_test = np.array([t.mean(axis=0) for t in Xtxt_test])
    n_test = Xim_test.shape[0]


    lr = 0.01
    l2 = 0.
    K = 5
    
    model = Multimodal.create(dim_img, dim_txt, dim_multi, l2, pooling, optim)
#    
    fwrite('Start training\n')
    sys.stdout.flush()
    break_all = False
    n_updates = 0
    for epoch in np.arange(51):
        train_idxs = np.arange(n_train)
        np.random.shuffle(train_idxs)
        product_ids_train = product_ids_train[train_idxs]
        Xim_train = Xim_train[train_idxs]
        Xtxt_train = Xtxt_train[train_idxs]
        fwrite('Epoch: %d\n' % epoch)
        
#==============================================================================
# Test        
#==============================================================================
        
        if not epoch%5:
            W_txt = model.W_txt.get_value()
            W_img = model.W_img.get_value()
            Xim_test_m = np.dot(Xim_test, W_img)
            
            Xtxt_test_m = np.dot(Xtxt_test, W_txt)
            S = cosine_similarity(Xim_test_m, Xtxt_test_m)
            
            args = np.argsort(-S, axis=1)
            test_ranks = [(arg == idx).nonzero()[0][0] for idx, arg in enumerate(args)]
              
            d = S.diagonal()
            a = args[:,0]
            s = S[np.arange(n_test), a]
            
            
            plt.figure(figsize=(15,15))
            plt.title('Rank histogram epoch %d'% epoch)
            sns.distplot(test_ranks, label='Rank of the correct label (median = %d)' % np.median(test_ranks))
            plt.legend()
            plt.savefig(os.path.join(folder, 'rank_%d' %epoch))
            
            
            plt.figure(figsize=(15,15))
            plt.xlim([-0.1, 1.1])
            plt.title('Similarities epoch %d' % epoch)
            sns.distplot(d, label='Correct (im,txt)')
            sns.distplot(s, label='Best (im,txt)')
            plt.legend()
            plt.savefig(os.path.join(folder, 'sims_%d' %epoch))
            
            median_rank = np.median(test_ranks)
            fwrite('\tTest median rank = %d\n' % median_rank)
            sys.stdout.flush()
#==============================================================================
# Train
#==============================================================================
        tic = time()
        for image, text_p in zip(Xim_train, Xtxt_train):
            d_neg = np.random.choice(n_train, size=K, replace=False)
            xtxt_n = text_embeddings[d_neg]
            n_updates += 1
            cost = model.train(image, text_p, xtxt_n, lr)
            if np.isnan(cost):
                fwrite('\n NAN\n')
                break_all = True
            if break_all:
                break
        toc = time() - tic
        model.save(saving_path)
        fwrite('\tTime = %fs\n' % toc)
        if break_all:
            break