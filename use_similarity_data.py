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
fwrite = sys.stdout.write
dtype = 'float32'

        
if __name__=='__main__':
    np.random.seed(1234)
    
    
    data = np.load('r_similarity_data').all()
    folder = 'plots/r_sim'
    saving_path = os.path.join(folder, "model.mod")
    
    d = data[data.keys()[0]]
    dim_img = d['image_emb'].shape[0]
    dim_txt = d['text_emb'].shape[1]
    dim_multi = 150
    
    n_train = int(0.66*len(data))
    
    image_embeddings = []
    text_embeddings = []
    product_ids = []
    fwrite('Loading data ...\n')
    for d in data:
        if not len(product_ids)%5000:
            fwrite('%d\n' % len(product_ids))
        product_ids.append(d)
        text_embeddings.append(data[d]['text_emb'].astype(dtype))
        image_embeddings.append(data[d]['image_emb'].astype(dtype))
    del data

    fwrite('Done\n')
    product_ids = np.array(product_ids)
    image_embeddings = np.array(image_embeddings)
    text_embeddings = np.array(text_embeddings)
    indexes = np.arange(len(product_ids))
    np.random.shuffle(indexes)
    
    image_embeddings = image_embeddings[indexes]
    text_embeddings = text_embeddings[indexes]
    product_ids = product_ids[indexes]
    
    Xim_train = image_embeddings[:n_train]
    Xim_test = image_embeddings[n_train+1:]
    Xtxt_train = text_embeddings[:n_train]
    Xtxt_test = text_embeddings[n_train+1:]
    product_ids_train = product_ids[:n_train]
    product_ids_test = product_ids[n_train + 1 :]
    
    test_lengths = [x.shape[0] for x in Xtxt_test]
    test_ends = np.cumsum(test_lengths)
    Xtxt_test_c = np.concatenate(Xtxt_test)
    n_test = Xim_test.shape[0]
    lr = 0.01
    l2 = 0.
    K = 5
    model = Multimodal.create(dim_img, dim_txt, dim_multi, l2)
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
            
            Xtxt_test_m = np.dot(Xtxt_test_c, W_txt)
            sims = cosine_similarity(Xim_test_m, Xtxt_test_m)
            b = 0
            S = []
            for e in test_ends:
                s = sims[:,b:e].max(axis=1)
                b = e
                S.append(s)
            S = np.array(S)  
            
            args = np.argsort(-S, axis=1)
            test_ranks = [(arg == idx).nonzero()[0][0] for idx, arg in enumerate(args)]
              
            d = S.diagonal()
            a = args[:,0]
            s = S[np.arange(n_test), a]
            
            
            plt.figure(figsize=(15,15))
            plt.title('Rank histogram epoch %d'% epoch)
            sns.distplot(test_ranks, label='Rank of the correct label (median = %d)' % np.median(test_ranks))
            plt.legend()
            plt.show()
            plt.savefig(os.path.join(folder, 'rank_%d' %epoch))
            
            
            plt.figure(figsize=(15,15))
            plt.xlim([-0.1, 1.1])
            plt.title('Similarities epoch %d' % epoch)
            sns.distplot(d, label='Correct (im,txt)')
            sns.distplot(s, label='Best (im,txt)')
            plt.legend()
            plt.show()
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