# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:00:43 2016

@author: hedi
"""
import numpy as np
import sys
import theano
import theano.tensor as T
import cPickle
fwrite = sys.stdout.write
dtype = 'float32'

epsilon = 1e-6
class Multimodal(object):
    @classmethod
    def create(cls, n_img, n_txt, n_hid, l2=None):
        W_img_val = 0.01*np.random.randn(n_img, n_hid).astype(dtype)
        W_txt_val = 0.01*np.random.randn(n_txt, n_hid).astype(dtype)
        return cls(W_img_val, W_txt_val, l2)
    
    @classmethod
    def load(cls,path):
        W_img_val, W_txt_val, l2 = cPickle.load(open(path,'r'))
        return cls(W_img_val, W_txt_val, l2)
        
    def save(self, path):
        to_save = [self.W_img.get_value(), self.W_txt.get_value(), self.l2]
        cPickle.dump(to_save, open(path,'w'))
        return None
        
    def __init__(self,W_img_val, W_txt_val, l2):
        self.l2 = l2
        self.W_img = theano.shared(W_img_val, name='W_img')
        self.W_txt = theano.shared(W_txt_val, name='W_txt')
        self.params = [self.W_img, self.W_txt]
        
        self.image = T.fvector('image')
        self.emb_image = T.dot(self.image, self.W_img)
        
        self.text_p = T.fmatrix('text_p')
        self.emb_text_p = T.dot(self.text_p, self.W_txt)
        
        self.text_n = T.ftensor3('text_n')
        self.emb_text_n = T.dot(self.text_n, self.W_txt)
        
        
        def sim_2(x, y):
            x_r = x / (epsilon + x.norm(2))
            y_r = y / (epsilon + y.norm(2, axis=0))
            return T.dot(x_r, y_r)
        
        def sim_3(x,y):
            x_r = x /  x.norm(2)
            norms = y.norm(2, axis=2)
            norms = T.reshape(norms, (norms.shape[0], norms.shape[1], 1))
            y_r = y / (epsilon + norms)
            return T.dot(y_r,x_r)
            
        self.sim_p = T.max(sim_2(self.emb_image, self.emb_text_p.T))
        self.sim3 = sim_3(self.emb_image, self.emb_text_n)
        self.sim_n = T.max(self.sim3, axis=1)
        self.maximum = T.maximum(0, 0.5 + self.sim_n - self.sim_p)
        self.cost = T.sum(self.maximum)
        
        if self.l2:
            for p in self.params:
                self.cost += (T.sum(p**2))
#        
        self.lr = T.fscalar('lr')
        self.grads = T.grad(self.cost, self.params)
        self.updates = [(p, p-self.lr*g) for p,g in zip(self.params, self.grads)]

        self._train = theano.function(inputs=[self.image, 
                                              self.text_p, 
                                              self.text_n, 
                                              self.lr],
                                      outputs = self.cost, 
                                      updates = self.updates,
                                      allow_input_downcast=True)
#        
#        # Test part
        self._test = theano.function(inputs=[self.image, self.text_p],
                                     outputs = self.sim_p,
                                     allow_input_downcast=True)
#
#
#    
    def train(self, image, text_p, xtxt_n, lr):
        lengths = [x.shape[0] for x in xtxt_n]
        m = max(lengths)
        text_n = np.array([np.pad(x, ((0,m-x.shape[0]), 
                                      (0,0)), mode='mean') for x in xtxt_n])
        return self._train(image, text_p, text_n, lr)
    
    def test(self, image, bag_of_vectors):
        return self._test(image, bag_of_vectors)