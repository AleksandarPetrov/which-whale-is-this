# -*- coding: utf-8 -*-
"""
Created on Tue May  8 09:48:21 2018

@author: Kuba
"""
from os import listdir
from os.path import splitext


def gen_imageName_dict(test_dir,train_dir, validation_percentage, file_names_sub = None):
    partition = {}
    test_list = listdir(test_dir)
    for i, filename in enumerate(test_list):
        test_list[i] = splitext(filename)[0]
        partition['test'] = test_list

    #Define the train list based on all data or n whales
    if file_names_sub == None:
        train_list = listdir(train_dir)
        for i,filename in enumerate(train_list):
            train_list[i] = splitext(filename)[0]
    else:
        train_list = file_names_sub

    len_train_list = len(train_list)
    len_val_set = int(validation_percentage * len_train_list)
    len_train_set = len_train_list - len_val_set

    train_set = train_list[0:len_train_set]
    val_set = train_list[len_train_set:len_train_list]
    partition['train'] = train_set
    partition['validation'] = val_set

    return partition


