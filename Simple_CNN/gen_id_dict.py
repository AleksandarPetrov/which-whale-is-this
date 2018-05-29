# -*- coding: utf-8 -*-
"""
Created on Tue May  8 09:48:21 2018
 
@author: Kuba
"""
from os import listdir
from os.path import splitext
 
def gen_id_dict(test_dir,train_dir):
    test_list = listdir(test_dir)
    train_list = listdir(train_dir)
    
    for i,filename in enumerate(test_list):
        test_list[i] = splitext(filename)[0]
    for i,filename in enumerate(train_list):
        train_list[i] = splitext(filename)[0]        
    partition = {}
    partition['train'] = train_list
    partition['test'] = test_list
    
    return partition
    
