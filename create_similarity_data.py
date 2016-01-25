# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 11:26:11 2016

@author: hedi

We create a file containing representation of images and texts. This file 
uses the img_caffe_features.dat file, where each line is as follows :

<path_to_image> <embedding>

(the path_to_image is incorrect, one must only keep the picture name)
"""

import numpy as np
import sys
import json
from goldberg.scripts.infer import Embeddings
from tokenizer import word_tokenize
from os import walk
from os.path import join
fwrite = sys.stdout.write
def text_embedding(product):
    vec = []
    for W in product.values():
        for w in W:
            try:
                w_v = model.word2vec(w)
            except:
                continue
            vec += [w_v]
    return np.array(vec)
    
if __name__ == '__main__':
    which_set = 't'
    csv_file = open(which_set+'_img_caffe_features.dat')
    products = {}
    fwrite('Going through image features dataset... \n')
    for line in csv_file:
        L = line.split(';')
        idx = L[0].split('/')[-1].split('.')[0]
        image_emb = np.array([float(l) for l in L[1].strip().split(' ')])
        products[idx] = {'image_emb':image_emb}
    fwrite('Done\n')
    csv_file.close()
    csv_file = open('training.csv', 'r')
    all_keys = csv_file.readline().split(';')
    product_keys = ['Description', 'Libelle','Marque']
    sys.stdout.flush()
    model = Embeddings('../product2vec2/embeddings/all/vecs.npy')
    dimension = model._vecs.shape[1]
    n_max = len(products)
    j = 0
    fwrite('Going through products to extract text embeddings... \n')
    for line in csv_file:
        j += 1
        if not j % 100000:
            fwrite('\t%d\n' % j)
            sys.stdout.flush()
        L = line.lower().split(';')
        idx = L[0]
        if idx in products:
            raw_product = dict([(k,v.decode('utf-8')) for k,v in zip(all_keys, L)])
            product = dict([(k,word_tokenize(raw_product[k])) for k in product_keys])
            vecs = text_embedding(product)
            products[idx]["text_emb"] = vecs
            products[idx]['product'] = json.dumps(raw_product)    
        
        
    fwrite('Done\n')
    images_path = 'images/img/training'
    break_all = False
    K = 0
    fwrite('Retrieving image paths... ')
    for (dirpath, dirnames, filenames) in walk(images_path):
        for f in filenames:
            idx = f.split('.')[0]
            if idx in products:
                products[idx]['image_path'] = join(dirpath,f)
                K += 1
            if K == len(products):
                break_all = True
            if break_all:
                break
        if break_all:
            break
    fwrite('Done\n')
    
    
    np.save(open(which_set+'_similarity_data','w'), products)
