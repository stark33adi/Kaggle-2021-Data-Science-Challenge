#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 19:33:26 2021

@author: peeyushkumar
"""

from DataReader import read_and_manipulate
import pandas as pd


''' NOTE : the NA in the csv can be read as nan by pandas so convert dtype to str while loading'''


''' the below class wll call the preprocessing module in its constructor'''
class data_pre_processor:
    
    def __init__(self,path,to_preprocess=True,load_saved=False,saved_name=None):

        if to_preprocess==True:
            read_and_manipulate(path).Driver(saving_path='preprocessed_data.csv')
            self.data=pd.read_csv('preprocessed_data.csv').astype('str')
            
        if load_saved == True:
            self.data=pd.read_csv(path).astype('str')
            
        self.labels=self.data.columns.values
        self.data=self.data.drop(index=0,columns=self.labels[0])
        self.labels=self.data.columns.values
        
    def encode_categories(self):
        
        data=self.data
        for col_names in self.labels[1:]:
            data[col_names]=data[col_names].astype('category').cat.codes
        
        return data
    
    def Driver(self,saving_path=None):

        data=self.encode_categories()
        if saving_path!=None:
            data.to_csv(saving_path)
        else:
            data.to_csv('processed_and_encoded_survey.csv')
        


