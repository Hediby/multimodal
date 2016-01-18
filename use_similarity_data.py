# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 11:54:09 2016

@author: hedi
"""
import numpy as np
import sys
import theano
import theano.tensor as T
fwrite = sys.stdout.write
dtype = 'float32'

class Multimodal(object):
    def __init__(self,W_img_val, W_txt_val):
        self.W_img = theano.shared(W_img_val, name='W_img')
        self.W_txt = theano.shared(W_txt_val, name='W_txt')
        self.params = [self.W_img, self.W_txt]
        
        self.image = T.fvector('image')
        self.text_p = T.fmatrix('text_p')
        self.text_n = T.fmatrix('text_n')
        self.lr = T.fscalar('lr')
        
        self.emb_image = T.dot(self.image, self.W_img)
        self.emb_text_n = T.dot(self.text_n, self.W_txt)
        self.emb_text_p = T.dot(self.text_p, self.W_txt)
        
        def sim(x, y):
            x_r = x / x.norm(2)
            y_r = y / y.norm(2, axis=0)
            return T.dot(x_r, y_r)
        
        self.sim_p = T.max(sim(self.emb_image, self.emb_text_p.T))
        self.sim_n = T.max(sim(self.emb_image, self.emb_text_n.T))
        
        self.cost = T.maximum(0, .5 - self.sim_p + self.sim_n)
        self.grads = T.grad(self.cost, self.params)
        self.updates = [(p, p-self.lr*g) for p,g in zip(self.params, self.grads)]
        
        self._train = theano.function(inputs=[self.image, 
                                              self.text_p, 
                                              self.text_n, 
                                              self.lr],
                                      outputs = self.cost, 
                                      updates = self.updates,
                                      allow_input_downcast=True)
                                      
        
        # Test part
        self._test = theano.function(inputs=[self.image, self.text_p],
                                     outputs = self.sim_p,
                                     allow_input_downcast=True)
                                     
    @classmethod
    def create(cls, n_img, n_txt, n_hid):
        W_img_val = 0.01*np.random.randn(n_img, n_hid).astype(dtype)
        W_txt_val = 0.01*np.random.randn(n_txt, n_hid).astype(dtype)
        return cls(W_img_val, W_txt_val)
    
    def train(self, xim, xtxt, xtxt_n, lr):
        return self._train(xim, xtxt, xtxt_n, lr)
    
    def test(self, image, bag_of_vectors):
        return self._test(image, bag_of_vectors)
        
        
if __name__=='__main__':
    np.random.seed(1234)
    
    data = np.load('similarity_data').all()
    
    d = data[data.keys()[0]]
    dim_img = d['image_emb'].shape[0]
    dim_txt = d['text_emb'].shape[1]
    dim_multi = 150
    
    n_train = int(0.66*len(data))
    
    image_embeddings = []
    text_embeddings = [] 
    for d in data.itervalues():
        text_embeddings.append(d['text_emb'])
        image_embeddings.append(d['image_emb'])
    product_ids = np.array(data.keys())
    image_embeddings = np.array(image_embeddings)
    text_embeddings = np.array(text_embeddings)
    
    indexes = np.arange(len(data))
    np.random.shuffle(indexes)
    
    image_embeddings = image_embeddings[indexes]
    text_embeddings = text_embeddings[indexes]
    product_ids = product_ids[indexes]
    
    Xim_train = image_embeddings[:n_train]
    Xim_test = image_embeddings[n_train+1:]
    Xtxt_train = text_embeddings[:n_train]
    Xtxt_test = text_embeddings[n_train+1:]
    product_ids_train = product_ids[:n_train]
    product_ids_test = product_ids[n_train + 1 : ]
    
    lr = 0.001
    model = Multimodal.create(dim_img, dim_txt, dim_multi)
    for epoch in np.arange(50):
        train_idxs = np.arange(n_train)
        np.random.shuffle(train_idxs)
        product_ids_train = product_ids_train[train_idxs]
        Xim_train = Xim_train[train_idxs]
        Xtxt_train = Xtxt_train[train_idxs]        
        if not epoch%5:
            train_sims = []
            for xim, xtxt in zip(Xim_train, Xtxt_train):
                train_sims.append(model.test(xim, xtxt))
            test_sims = []
            for xim, xtxt in zip(Xim_test, Xtxt_test):
                test_sims.append(model.test(xim, xtxt))
            fwrite('Epoch: %d\n' % epoch)
            fwrite('\tTrain score = %f\n' % np.mean(train_sims))
            fwrite('\tTest score = %f\n' % np.mean(test_sims))
            sys.stdout.flush()
            
        for xim, xtxt in zip(Xim_train, Xtxt_train):
            d_neg = np.random.choice(n_train)
            xtxt_n = text_embeddings[d_neg]
            model.train(xim, xtxt, xtxt_n, lr)