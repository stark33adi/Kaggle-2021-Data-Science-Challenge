#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 14:10:37 2021

@author: adityakumar
"""

import pandas as pd
import pickle as pkl

''' N : not choosen, NA : not applicable'''

class read_and_manipulate:
    
    def __init__(self,path,nan_replacenment_dict=None,initial_nan_filling=True):
        
        ''' this constructor will load the dataset '''
        
        self.path=path
        
        try:
            self.data=pd.read_csv(path)
        except:
            print('faulty path')
        
        self.total_columns=len(self.data.columns)
        self.total_rows=len(self.data)
        self.labels_name=self.data.columns.values
        
        if nan_replacenment_dict != None:
            with open(nan_replacenment_dict, 'rb') as f:
                self.nan_replacements = pkl.load(f)
                
        # lets fill every null value as N (this step can be over ridden with the flag : initial_nan_filling) 
        if initial_nan_filling == True:
            self.data=self.data.fillna('N')

            
    ''' this function will replace null values in specific column with specific values '''
    
    def null_manipulator(self,column_name,to_replace_with=None): 
            
            
        if to_replace_with==None:
            return 'Please Enter the Entity to replace missing values with'
        
        else:
            self.data[column_name].fillna(to_replace_with)
            
            
    def fill_null(self,default_replacenment_dict=True):
        
        
        if (default_replacenment_dict==False):
            path=input('Please paste the path for the replacement dict : ')
            with open(str(path), 'rb') as f:
                replacement_dict = pkl.load(f)
        else:
            replacement_dict=self.nan_replacenment_dict
            
        for key,value in replacement_dict.item():
            self.null_manipulator(key,value)
            
            
    ''' below function act as a UI for manually editing null values in each of the column'''
    def enter_custom_values_to_replace_nan(self):
        
        dict={}
        print('\n')
        print('You have to select value to replace nan of a specific column with specific value, the option are as follows : ')
        print('Enter ''end'' to end the process')
        print('Enter the value you want to replace nan with for that column')
        print('Enter "skip" for skip that column')
        print('\n')
        print('and please be consistent with the entries')
            
        i=0
        while True:
            
            val = input('Enter replace value for "'+self.labels_name[i]+'" : ')
            if str(val).lower() == 'end':
                    break
            elif str(val).lower() == 'skip':
                    continue
            else:   
                    dict[self.labels_name[i]]=str(val).lower()
            i=i+1
        
        val = input('do you want to save the inputs : Y/N ')
        if str(val).upper()=='Y':   
             with open('nan_replacements.pkl', 'wb') as f:
                    pkl.dump(dict, f)
             print('Saved!')
        else:
            pass
             
        
        
    def save_data(self,name=None):
        if name!=None:
            self.data.to_csv(name)
        else:
            self.data.to_csv('Test.csv')
        
        
        ''' this functions below will handle special question'''
        
    def handle_Q_18_19(self):
        
        ''' Question 18 and 19 were asked to candidates who answered options 
        other than 'None' for Question 17 '''
        ''' question 17 is from column 90 to 101 inclusive. i.e [90,102] '''
        ''' "None" and "Other" columns lies at 100 and 101 respectively '''
        
        ''' Objective: to add column before 18 and 19 which states whether the Question
        was asked to the user or not'''
        
        entries=[]
        for none,other in zip(self.data[self.labels_name[100]],self.data[self.labels_name[101]]):
            if none == 'None':
                entries.append(0)
            else:
                entries.append(1)
        
        self.data.insert(102, "Question 18 & 19 asked or not ?", entries, True)
        
        
        h=self.data.columns.values
        ''' question 18 and 19 is from indexes 103 to 115 inclusive.'''
        target=h[103:116]
        for i in range(1,len(self.data)):
            if entries[i]==0:
                for cols in target:
                    self.data[cols][i]='NA'

         
           
    def handle_Q_28_29_30(self):
        
        ''' Question 28 were asked to candidates who answered 2 or more options 
        for Question 27_A given that the did not answer "None" '''
        ''' question 27_A is from column 130 to 141 inclusive. i.e [130,142] '''
        ''' question 28 is on column 142 '''
        ''' "None" and "Other" columns for 27_A lies at 140 and 141 respectively '''
        ''' Objective: to add column before Q 28 which states whether the Question
        was asked to the user or not'''
        
        
        ''' Question 29 and 30_A were asked to candidates who did not answer "None" '''
        ''' question 29 is from column 143 to 147 inclusive'''
        ''' question 30 is from column 148 to 155 inclusive'''
        
        ''' "None" and "Other" columns for 27_A lies at 140 and 141 respectively '''
        ''' Objective: to add column before Q 28 which states whether the Question
        was asked to the user or not'''
        
        entries=[0]
        entries_29_30=[0]
        labels=self.data.columns.values
        
        for i in range(1,len(self.data)):
            
            if self.data[labels[140]][i]=='None':
                entries.append(0)
                entries_29_30.append(0)
                self.data[labels[142]][i]='NA'
                for cols in labels[143:156]:
                    self.data[cols][i]='NA'
            else:
                entries_29_30.append(1)
                temp=0
                for j in labels[130:142]:
                    if j !=labels[140] and self.data[j][i]!='N':
                        temp=temp+1
                        if(temp>=2):
                            break
                if temp>=2:
                    entries.append(1)
                else:
                    entries.append(0)
                    self.data[labels[142]][i]='NA'
                    
        self.data.insert(142, "Question 28 asked or not ?", entries, True)
        self.data.insert(144, "Question 29_A and 30_A asked or not ?", entries_29_30, True)
        
        
    def handle_Q_33(self):
        
        ''' Question 33 were asked to candidates who answered 2 or more options 
        for Question 32_A given that the did not answer "None" '''
        ''' question 33 is on column 189'''
        ''' question 32_A is from column 168 to 188 inclusive. i.e [168,189] '''
        ''' "None" and "Other" columns for 32_A lies at 187 and 188 respectively '''
        ''' Objective: to add column before Q 33 which states whether the Question
        was asked to the user or not'''

        entries=[0]
        labels=self.data.columns.values
        
        for i in range(1,len(self.data)):
            
            if self.data[labels[187]][i]=='None':
                entries.append(0)
                self.data[labels[189]][i]='NA'      
            else:
                temp=0
                for j in labels[168:189]:
                    if j !=labels[187] and self.data[j][i]!='N':
                        temp=temp+1
                        if(temp>=2):
                            break
                if temp>=2:
                    entries.append(1)
                else:
                    entries.append(0)
                    self.data[labels[189]][i]='NA'
                    
        self.data.insert(189, "Question 33 asked or not ?", entries, True)
        
        
    def handle_Q_35(self):
        
        ''' Question 35 were asked to candidates who answered 2 or more options 
        for Question 34_A given that the did not answer "None" '''
        
        ''' question 35 is on column 208'''
        ''' question 34_A is from column 191 to 207 inclusive. i.e [191,208] '''
        ''' "None" and "Other" columns for 32_A lies at 206 and 207 respectively '''
        ''' Objective: to add column before Q 35 which states whether the Question
        was asked to the user or not'''

        entries=[0]
        labels=self.data.columns.values
        for i in range(1,len(self.data)):
            
            if self.data[labels[206]][i]=='None':
                entries.append(0)
                self.data[labels[208]][i]='NA'      
            else:
                temp=0
                for j in labels[191:208]:
                    if j !=labels[206] and self.data[j][i]!='N':
                        temp=temp+1
                        if(temp>=2):
                            break
                if temp>=2:
                    entries.append(1)
                else:
                    entries.append(0)
                    self.data[labels[208]][i]='NA'
                    
        self.data.insert(208, "Question 35 asked or not ?", entries, True)
    
    def handle_37_A(self):
        
        ''' Question 37_A was asked to candidates who answered options 
        other than 'None' for Question 36_A '''
        ''' question 36_A is from column 210 to 217 inclusive. i.e [210,218] '''
        ''' "None" and "Other" columns lies at 216 and 217 respectively '''
        
        ''' Objective: to add column before 37_A which states whether the Question
        was asked to the user or not'''
        ''' Question 37_A is from 218 to 225 inclusive'''
        
        h=self.data.columns.values
        entries=[]
        i=0
       
        for none in self.data[h[216]]:
            if none ==  'No / None':
                entries.append(0)
                for col_name in h[218:226]:
                    self.data[col_name][i]='NA'
            else:
                entries.append(1)
            i=i+1
        
        
        self.data.insert(218, "Question 37_A asked or not ?", entries, True)
        
    def handle_supplemental_questions(self):
        
        ''' Supplemental questions were asked to students, unemployed 
        , or those who never spend any money on cloud '''
        ''' "Student" and "Currently not employed" are option from Q5'''
        ''' Q5 is on column 5 '''
        ''' People who havent spend any money on cloud "$0 ($USD)" can be identified
        from Q26'''
        ''' Q26 is on column 129 '''
        '''Supplement questions start from column [274: ] '''
        
        ''' objective 1: is to put NA in 'orig_ques' for students,unemployed'''
        
        
        labels=self.data.columns.values
        
        orig_ques=['Q27_A_Part_1', 'Q27_A_Part_2', 'Q27_A_Part_3', 'Q27_A_Part_4', 'Q27_A_Part_5', 
                   'Q27_A_Part_6', 'Q27_A_Part_7', 'Q27_A_Part_8', 'Q27_A_Part_9', 'Q27_A_Part_10',
                   'Q27_A_Part_11', 'Q27_A_OTHER', 'Q29_A_Part_1', 'Q29_A_Part_2', 'Q29_A_Part_3', 
                   'Q29_A_Part_4', 'Q29_A_OTHER', 'Q30_A_Part_1', 'Q30_A_Part_2', 'Q30_A_Part_3', 
                   'Q30_A_Part_4', 'Q30_A_Part_5', 'Q30_A_Part_6', 'Q30_A_Part_7', 'Q30_A_OTHER', 
                   'Q31_A_Part_1', 'Q31_A_Part_2', 'Q31_A_Part_3', 'Q31_A_Part_4', 'Q31_A_Part_5',
                   'Q31_A_Part_6', 'Q31_A_Part_7', 'Q31_A_Part_8', 'Q31_A_Part_9', 'Q31_A_OTHER', 
                   'Q32_A_Part_1', 'Q32_A_Part_2', 'Q32_A_Part_3', 'Q32_A_Part_4', 'Q32_A_Part_5',
                   'Q32_A_Part_6', 'Q32_A_Part_7', 'Q32_A_Part_8', 'Q32_A_Part_9', 'Q32_A_Part_10',
                   'Q32_A_Part_11', 'Q32_A_Part_12', 'Q32_A_Part_13', 'Q32_A_Part_14', 'Q32_A_Part_15',
                   'Q32_A_Part_16', 'Q32_A_Part_17', 'Q32_A_Part_18', 'Q32_A_Part_19', 'Q32_A_Part_20', 
                   'Q32_A_OTHER', 'Q34_A_Part_1', 'Q34_A_Part_2', 'Q34_A_Part_3', 'Q34_A_Part_4', 
                   'Q34_A_Part_5', 'Q34_A_Part_6', 'Q34_A_Part_7', 'Q34_A_Part_8', 'Q34_A_Part_9', 
                   'Q34_A_Part_10', 'Q34_A_Part_11', 'Q34_A_Part_12', 'Q34_A_Part_13', 'Q34_A_Part_14',
                   'Q34_A_Part_15', 'Q34_A_Part_16', 'Q34_A_OTHER', 'Q36_A_Part_1', 'Q36_A_Part_2',
                   'Q36_A_Part_3', 'Q36_A_Part_4', 'Q36_A_Part_5', 'Q36_A_Part_6', 'Q36_A_Part_7', 
                   'Q36_A_OTHER', 'Q37_A_Part_1', 'Q37_A_Part_2', 'Q37_A_Part_3', 'Q37_A_Part_4', 
                   'Q37_A_Part_5', 'Q37_A_Part_6', 'Q37_A_Part_7', 'Q37_A_OTHER', 'Q38_A_Part_1', 
                   'Q38_A_Part_2', 'Q38_A_Part_3', 'Q38_A_Part_4', 'Q38_A_Part_5', 'Q38_A_Part_6',
                   'Q38_A_Part_7', 'Q38_A_Part_8', 'Q38_A_Part_9', 'Q38_A_Part_10', 'Q38_A_Part_11', 
                   'Q38_A_OTHER']
        
        #supp_ques=labels[274:]
        
        for i in range(1,len(self.data)):
            
            if self.data[labels[5]][i]=='Student' or self.data[labels[5]][i]=='Currently not employed' or self.data[labels[129]][i]=='$0 ($USD)':
                for col_name in orig_ques:
                    self.data[col_name][i]='NA'

        return labels
    
    def handle_special_conditions(self):
        
        ''' "Student" and "Currently not employed" are option from Q5'''
        ''' Q5 is on column 5 '''
        
        '''Special condition 1 : those who have chosen that they have never coded in their life were not asked questions about coding'''
        '''Objective 1 : Remove Examples that satisfy special condition 1'''
        
        
        '''Special Condition 2 : Students and Un employed were not asked questions about their employer'''
        ''' Questions related to employment : Q 20,21,22,23,24,25  index from [116: 129] '''
        '''Objective 2 : put NA in employers question for students and unemployed'''
        
        employer_ques=['Q20', 'Q21', 'Q22', 'Q23', 'Q24_Part_1', 'Q24_Part_2',
       'Q24_Part_3', 'Q24_Part_4', 'Q24_Part_5', 'Q24_Part_6',
       'Q24_Part_7', 'Q24_OTHER', 'Q25']
        
        labels=self.data.columns.values
        
        no_coding_indexes=[]
        
        for i in range(1,len(self.data)):
            
            if self.data[labels[5]][i]=='Student' or self.data[labels[5]][i]=='Currently not employed' :
                for col_name in employer_ques:
                    self.data[col_name][i]='NA'
            
            if self.data[labels[6]][i]=='I have never written code':
                no_coding_indexes.append(i)
                    
        self.data=self.data.drop(index = no_coding_indexes)

    
    def last_touchup(self):
        
        ''' In this function we will drop the extra columns we have added'''
        
        labels=self.data.columns.values
        labels_n=[102,142,144,189,208,218]
        extra_cols=[labels[x] for x in labels_n]
        self.data=self.data.drop(columns = extra_cols)
        print(self.data.isnull().sum().sum())
        

    def Driver(self,saving_path=None):
        
        
        self.handle_Q_18_19()
        self.handle_Q_28_29_30()
        self.handle_Q_33()
        self.handle_Q_35()
        self.handle_37_A()
        self.handle_supplemental_questions()
        self.handle_special_conditions()
        self.last_touchup()
        

        if saving_path!=None:
            self.save_data(name=saving_path)
        else:
            self.save_data()

        
        

'''
path='/Users/peeyushkumar/Desktop/DS Project/kaggle-survey-2021/kaggle_survey_2021_responses.csv'

'''
