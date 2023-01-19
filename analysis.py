#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 23:55:53 2021

@author: peeyushkumar
"""



from sklearn import cluster , preprocessing,decomposition 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import pickle as pkl

class visualization:
    
    def __init__(self,path='processed_survey.csv'):
        
        self.data=pd.read_csv(path)
        self.labels=self.data.columns.values
        self.total_rows=len(self.data)
        
        
    def get_data(self):
        
        return self.data, self.labels
    
    def prof_vs_lang(self):
        
        ''' columne for professions is Q5'''
        data=self.data
        target_index=['Q7_Part_1', 'Q7_Part_2', 'Q7_Part_3',
       'Q7_Part_4', 'Q7_Part_5', 'Q7_Part_6', 'Q7_Part_7', 'Q7_Part_8',
       'Q7_Part_9', 'Q7_Part_10', 'Q7_Part_11']
        
        dict={'Q7_Part_1':'Python', 'Q7_Part_2':'R', 'Q7_Part_3':'SQL',
       'Q7_Part_4':'C', 'Q7_Part_5':'C++', 'Q7_Part_6':'Java', 'Q7_Part_7':'Javascript',
       'Q7_Part_8':'Julia','Q7_Part_9':'Swift', 'Q7_Part_10':'Bash', 'Q7_Part_11':'MATLAB'}
        
        repo=[]
        professions=[]

        for lang in target_index:
            temp={}
            for i in range(1,self.total_rows):
                if data[lang][i]!='N' and  data[lang][i]!='NA' :
                    
                    if data['Q5'][i] in temp.keys():
                        temp[data['Q5'][i]]=temp[data['Q5'][i]]+1
                    else:
                        temp[data['Q5'][i]]=1
                        
                    if data['Q5'][i] not in professions:
                        professions.append(data['Q5'][i])
                        
            repo.append(temp)
        i=0
        legends=[]
        for lang_dict in repo:
            
            x=[]
            y=[]
            for prof in professions:
                if prof in lang_dict.keys():
                    x.append(prof)
                    y.append(lang_dict[prof])
                else:
                    x.append(prof)
                    y.append(0)

            self.plot_graph(y,labelx='Professions-->',labely='Used by-->',
                            title='Profession vs Programming Language',x=x)   
            legends.append(dict[target_index[i]])
            i=i+1
        plt.legend(legends)
        plt.show()
        
    def prof_vs_salaries(self):
        
        ''' columne for professions is Q5'''
        
        data=self.data
        target_index=['Q25']
        repo=[]
        professions=[]
        labelx='Professions-->'
        labely='Frequency-->'
        title='Profession vs Salaries'
        
        lis=['100,000-124,999','70,000-79,999',
       '150,000-199,999', '80,000-89,999', '125,000-149,999',
       '90,000-99,999', '200,000-249,999', '300,000-499,999',
       '250,000-299,999', '>$1,000,000', '$500,000-999,999']

        for lang in target_index:
            for pay in lis:
                temp={}
                for i in range(1,self.total_rows):
                    if data[lang][i]!='N' and  data[lang][i]!='NA' :
                        
                        if data[lang][i]==pay:
                            if data['Q5'][i] in temp.keys():
                                temp[data['Q5'][i]]=temp[data['Q5'][i]]+1
                            else:
                                temp[data['Q5'][i]]=1
                                
                        if data['Q5'][i] not in professions:
                            professions.append(data['Q5'][i])
                            
                repo.append(temp)
        i=0
        legends=[]
        for lang_dict in repo:
            
            x=[]
            y=[]
            for prof in professions:
                if prof in lang_dict.keys():
                    x.append(prof)
                    y.append(lang_dict[prof])
                else:
                    x.append(prof)
                    y.append(0)

            self.plot_graph(y,labelx=labelx,labely=labely,
                            title=title,x=x)   
            
            legends.append(lis[i])
            
            i=i+1
        plt.legend(legends)
        plt.show()
        
    def prof_vs_degree(self):
        
        ''' columne for professions is Q5'''
        
        data=self.data
        target_index=['Q4']
        repo=[]
        professions=[]
        labelx='Professions-->'
        labely='Frequency-->'
        title='Profession vs Degree Type'
        
        lis=['Master’s degree', 'Bachelor’s degree', 'Doctoral degree',
       'Some college/university study without earning a bachelor’s degree'
       , 'No formal education past high school',
       'Professional doctorate']

        for lang in target_index:
            for pay in lis:
                temp={}
                for i in range(1,self.total_rows):
                    if data[lang][i]!='N' and  data[lang][i]!='NA' :   
                        
                        if data[lang][i]==pay:
                            
                            if data['Q5'][i] in temp.keys():
                                temp[data['Q5'][i]]=temp[data['Q5'][i]]+1
                            else:
                                temp[data['Q5'][i]]=1
                                
                        if data['Q5'][i] not in professions:
                            professions.append(data['Q5'][i])
                            
                repo.append(temp)
        i=0
        legends=[]
        for lang_dict in repo:
            
            x=[]
            y=[]
            for prof in professions:
                if prof in lang_dict.keys():
                    x.append(prof)
                    y.append(lang_dict[prof])
                else:
                    x.append(prof)
                    y.append(0)

            self.plot_graph(y,labelx=labelx,labely=labely,
                            title=title,x=x)   
            
            legends.append(lis[i])
            
            i=i+1
        plt.legend(['Master’s degree', 'Bachelor’s degree', 'Doctoral degree',
       'Some college/university'
       , 'No education past high school',
       'Professional doctorate'])
        plt.show()
        
    def salary_vs_degree(self):
        
        ''' columne for salary is Q5'''
        
        data=self.data
        target_index=['Q25']
        target_x='Q4'
        repo=[]
        professions=[]
        labelx='Degree-->'
        labely='Frequency-->'
        title='Salary vs Degree Type'
        
        lis=['100,000-124,999','70,000-79,999',
       '150,000-199,999', '80,000-89,999', '125,000-149,999',
       '90,000-99,999', '200,000-249,999', '300,000-499,999',
       '250,000-299,999', '>$1,000,000', '$500,000-999,999']

        for lang in target_index:
            for pay in lis:
                temp={}
                for i in range(1,self.total_rows):
                    if data[lang][i]!='N' and  data[lang][i]!='NA'and data['Q5'][i]!='Student'  and data['Q5'][i]!='Currently not employed' :
                        
                        if data[lang][i]==pay:
                            
                            if data[target_x][i] in temp.keys():
                                temp[data[target_x][i]]=temp[data[target_x][i]]+1
                            else:
                                temp[data[target_x][i]]=1
                                
                        if data[target_x][i] not in professions:
                            professions.append(data[target_x][i])
                            
                repo.append(temp)
        i=0
        legends=[]
        for lang_dict in repo:
            
            x=[]
            y=[]
            for prof in professions:
                if prof in lang_dict.keys():
                    x.append(prof)
                    y.append(lang_dict[prof])
                else:
                    x.append(prof)
                    y.append(0)

            self.plot_graph(y,labelx=labelx,labely=labely,
                            title=title,x=x)   
            
            legends.append(lis[i])
            
            i=i+1
        plt.legend(legends)
        plt.show()
    
    def coding_exp_vs_earning(self):
        data=self.data  
        target_index=['Q25']
        target_x='Q6'
        repo=[]
        professions=[]
        labelx='Coding Experience-->'
        labely='Frequency-->'
        title='Coding Experience vs Salary'
        
        lis=['100,000-124,999','70,000-79,999',
       '150,000-199,999', '80,000-89,999', '125,000-149,999',
       '90,000-99,999', '200,000-249,999', '300,000-499,999',
       '250,000-299,999', '>$1,000,000', '$500,000-999,999']

        for lang in target_index:
            for pay in lis:
                temp={}
                for i in range(1,self.total_rows):
                    if data[lang][i]!='N' and  data[lang][i]!='NA'and data['Q5'][i]!='Student'  and data['Q5'][i]!='Currently not employed' :
                        
                        if data[lang][i]==pay:
                            
                            if data[target_x][i] in temp.keys():
                                temp[data[target_x][i]]=temp[data[target_x][i]]+1
                            else:
                                temp[data[target_x][i]]=1
                                
                        if data[target_x][i] not in professions:
                            professions.append(data[target_x][i])
                            
                repo.append(temp)
        i=0
        legends=[]
        for lang_dict in repo:
            
            x=[]
            y=[]
            for prof in professions:
                if prof in lang_dict.keys():
                    x.append(prof)
                    y.append(lang_dict[prof])
                else:
                    x.append(prof)
                    y.append(0)

            self.plot_graph(y,labelx=labelx,labely=labely,
                            title=title,x=x)   
            
            legends.append(lis[i])
            
            i=i+1
        plt.legend(legends,loc='upper left', bbox_to_anchor=(0.5, 1.05),
          ncol=2)
        plt.show()
    
    def salary_vs_company_size(self):
        
        ''' columne for salary is Q25'''
        
        data=self.data
        target_index=['Q25']
        target_x='Q21'
        repo=[]
        professions=[]
        labelx='Company Size-->'
        labely='Frequency-->'
        title='Salary vs Company Size'
        
        lis=['100,000-124,999','70,000-79,999',
       '150,000-199,999', '80,000-89,999', '125,000-149,999',
       '90,000-99,999', '200,000-249,999', '300,000-499,999',
       '250,000-299,999', '>$1,000,000', '$500,000-999,999']

        for lang in target_index:
            for pay in lis:
                temp={}
                for i in range(1,self.total_rows):
                    if data[lang][i]!='N' and  data[lang][i]!='NA'and data['Q5'][i]!='Student'  and data['Q5'][i]!='Currently not employed' :
                        
                        if data[lang][i]==pay:
                            
                            if data[target_x][i] in temp.keys():
                                temp[data[target_x][i]]=temp[data[target_x][i]]+1
                            else:
                                temp[data[target_x][i]]=1
                                
                        if data[target_x][i] not in professions:
                            professions.append(data[target_x][i])
                            
                repo.append(temp)
        i=0
        legends=[]
        for lang_dict in repo:
            
            x=[]
            y=[]
            for prof in professions:
                if prof in lang_dict.keys():
                    x.append(prof)
                    y.append(lang_dict[prof])
                else:
                    x.append(prof)
                    y.append(0)

            self.plot_graph(y,labelx=labelx,labely=labely,
                            title=title,x=x)   
            
            legends.append(lis[i])
            
            i=i+1
        plt.legend(legends,loc='upper left', bbox_to_anchor=(0.5, 1.05),
          ncol=2)
        plt.show()
        
    def gender_vs_profession(self):
        
        ''' columne for salary is Q25'''
        
        data=self.data
        target_index=['Q2']
        target_x='Q5'
        repo=[]
        professions=[]
        labelx='Profession-->'
        labely='Frequency-->'
        title='Gender vs Profession'
        
        lis=['Man', 'Woman']

        for lang in target_index:
            for pay in lis:
                temp={}
                for i in range(1,self.total_rows):
                    if data[lang][i]!='N' and  data[lang][i]!='NA'and data['Q5'][i]!='Student'  and data['Q5'][i]!='Currently not employed' :
                        
                        if data[lang][i]==pay:
                            
                            if data[target_x][i] in temp.keys():
                                temp[data[target_x][i]]=temp[data[target_x][i]]+1
                            else:
                                temp[data[target_x][i]]=1
                                
                        if data[target_x][i] not in professions:
                            professions.append(data[target_x][i])
                            
                repo.append(temp)
        i=0
        legends=[]
        for lang_dict in repo:
            
            x=[]
            y=[]
            for prof in professions:
                if prof in lang_dict.keys():
                    x.append(prof)
                    y.append(lang_dict[prof])
                else:
                    x.append(prof)
                    y.append(0)

            self.plot_graph(y,labelx=labelx,labely=labely,
                            title=title,x=x)   
            
            legends.append(lis[i])
            
            i=i+1
        plt.legend(legends)
        plt.show()
        
    def gender_vs_earning(self):
        
        ''' columne for salary is Q25'''
        
        data=self.data
        target_index=['Q2']
        target_x='Q25'
        repo=[]
        professions=[]
        labelx='Earning-->'
        labely='Frequency-->'
        title='Gender vs Salary'
        
        lis=['Man', 'Woman']

        for lang in target_index:
            for pay in lis:
                temp={}
                for i in range(1,self.total_rows):
                    if data[lang][i]!='N' and  data[lang][i]!='NA'and data['Q5'][i]!='Student'  and data['Q5'][i]!='Currently not employed' :
                        
                        if data[lang][i]==pay:
                            
                            if data[target_x][i] in temp.keys():
                                temp[data[target_x][i]]=temp[data[target_x][i]]+1
                            else:
                                temp[data[target_x][i]]=1
                                
                        if data[target_x][i] not in professions:
                            professions.append(data[target_x][i])
                            
                repo.append(temp)
        i=0
        legends=[]
        for lang_dict in repo:
            
            x=[]
            y=[]
            for prof in professions:
                if prof in lang_dict.keys():
                    x.append(prof)
                    y.append(lang_dict[prof])
                else:
                    x.append(prof)
                    y.append(0)

            self.plot_graph(y,labelx=labelx,labely=labely,
                            title=title,x=x)   
            
            legends.append(lis[i])
            
            i=i+1
        plt.legend(legends)
        plt.show()
        
    def plot_graph(self,y_1=list,labelx=str,labely=str,title=str,x=None,legends=list):
    
        if x==None:
            x=[]
            for f in range(0,len(y_1)):
                x.append(f)
        else:
            x=x
        plt.plot(x,y_1)
        plt.xticks(rotation = 90)
        plt.title(title)
        plt.xlabel(labelx)
        plt.ylabel(labely)
        #plt.show()
        
    def Driver(self):
        
        self.prof_vs_degree()
        self.prof_vs_salaries()
        self.prof_vs_lang()
        self.salary_vs_company_size()
        self.salary_vs_degree()
        self.coding_exp_vs_earning()
        self.gender_vs_earning()
        self.gender_vs_profession()
        
        

class algos:
    
    def __init__(self,path='processed_and_encoded_survey.csv',label_path='processed_survey.csv'):
        
        self.data=pd.read_csv(path)
        self.labels=self.data.columns.values
        self.label_data=pd.read_csv(label_path)
        self.total_rows=len(self.label_data)
        self.n_features=len(self.data.columns.values)
        
    def get_data(self):
        
        return self.data, self.labels,self.label_data
    
    ''' clustering algorithms'''
        
    def kmeans_clustering(self,X,random_state=0,n_clusters=10):
        
        kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=random_state).fit(X)
        
        return kmeans
    
    def dbscan(self,X,eps=2,min_samples=5):
        
        return cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    
    def optics(self,X,min_samples=2):
        
        return cluster.OPTICS(min_samples=min_samples).fit(X)

    def birch(self,X,n_clusters=None):
    
        return cluster.Birch(n_clusters=n_clusters).fit(X)

    def spectral_clustering(self,X,n_clusters=2,random_state=0):
        
        return cluster.SpectralClustering(n_clusters=n_clusters, assign_labels='discretize',random_state=random_state).fit(X)
    
    def Mean_shift_clustering(self,X):
        
        return cluster.MeanShift(bandwidth=2).fit(X)
    
    
    def normal_scaling(self,X):
        return preprocessing.normalize(X, norm='l2')
    
    '''Dimensionality Reduction '''
    
    def ipca(self,X,n_components=2):
        return decomposition.IncrementalPCA(n_components=n_components, batch_size=200).fit_transform(X)
    
    def plot_graph(self,data,clusters,labelx=str,labely=str,title=str):
        
        
        dict={}
        for clus ,point in zip(clusters,data):
          
            if str(clus) not in dict.keys():
                dict[str(clus)]=[point]
            else:
                dict[str(clus)].append(point)
                
        legend=[]
        for key in dict.keys():
            x=[]
            y=[]
            
            for tup in dict[key]:
                x.append(tup[0])
                y.append(tup[1])
            
            plt.scatter(x,y)
            legend.append(key)
            
       # plt.xticks(rotation = 90)
            plt.title(title)
            plt.xlabel(labelx)
            plt.ylabel(labely)
       # plt.legend(legend)
        plt.show()
        
        
        
    def Seperate_cluster_data(self,data,clusters,labelx=str,labely=str,title=str,animate_label=str):
        

        x_animate={}
        for i in range(clusters.shape[-1]):
            if str(clusters[i]) not in x_animate.keys():
                x_animate[str(clusters[i])]=[i]
            else:
                x_animate[str(clusters[i])].append(i)
                
                
                
        for key in x_animate.keys():
            temp=[]
            for k in x_animate.keys():
                if k !=key:
                    temp=temp+x_animate[k]
                    
            t_data=self.label_data.drop(index=temp)
            t_data.to_csv(str(key)+'_cluster.csv')
            
        return x_animate
   
    def create__train_test_data(self,feature_to_predict):
        
        train=[]
        for col in self.labels[1:]:
            if col!=feature_to_predict:
                train.append(col)
                
        X=self.data[train]
        Y=self.data[feature_to_predict]
        
        X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.33, random_state=42)
        
        return X_train, X_test, y_train, y_test
    

    def plot_graph_normal(self,y_1=list,labelx=str,labely=str,title=str,x=None,flag=False):
    
        if x==None:
            x=[]
            for f in range(0,len(y_1)):
                x.append(f)
        else:
            x=x
            
        plt.plot(x,y_1)
        plt.title(title)
        plt.xlabel(labelx)
        if flag==True:
             plt.xticks(rotation = 90)
        plt.ylabel(labely)
        plt.show()
        
    def try_different_features(self,list_of_features_to_predict):
        
        x=[]
        y=[]
        
        dict={}
        for feature in list_of_features_to_predict:
            x_train, x_test, y_train, y_test =self.create__train_test_data(feature)
            dt=tree.DecisionTreeClassifier().fit(x_train,y_train)
        
            y_pred=dt.predict(x_test)
            acc=accuracy_score(y_test,y_pred)
            dict[feature]=acc
            x.append(feature)
            y.append(acc)
            
        ''' plot graph for the features'''
        self.plot_graph_normal(y,labelx='Features predicted-->',labely='Accuracy for each feature we predicted using the whole dataset-->',
                        title='Feature predicted vs Accuracy of its prediction',x=x)
      
        ''' Save the dictionary of predictions'''
        with open('different_feature_predictions.pkl', 'wb') as f:
                    pkl.dump(dict, f)
                    
        return dict
            
    def plot_top_preds(self,threshold=0.7,flag=False):
        
        with open('different_feature_predictions.pkl', 'rb') as f:
               Dict = pkl.load(f)
               
             
        
        x=[]
        y=[]
        for key,values in Dict.items():
            if values>=threshold:
                x.append(key)
                y.append(values)
        
           
        if flag==True:
            return len(x)
        
        print('Total Features which are above threshold of {} are {}.'.format(threshold,len(x)))
        

        self.plot_graph_normal(y,labelx='Type of Features predicted-->',labely='Accuracy for each feature we predicted using the whole dataset-->',
                        title='Feature predicted vs Accuracy of its prediction',x=None,flag=True)

                
    def calculate_dependancy_matrix_preporcessed(self):
        
        acc=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        
        values=[]
        for thr in acc:
            values.append(float(self.plot_top_preds(threshold=thr,flag=True)/self.n_features))
        
        return values,acc
        
        
               
        
    def Driver(self):
        
        #self.plot_top_preds(threshold=0.7)
        
        ''' scale data'''
        scale_data=self.normal_scaling(self.data[self.labels[2:]])
        
        ''' Reduce Dimensionality of Data'''
        r_data=self.ipca(scale_data)
        

        '''Plot kmeans cluster'''
        k=self.kmeans_clustering(scale_data,random_state=0,n_clusters=10)
        self.plot_graph(r_data,k.labels_,labelx='pc1',labely='pc2',title='Reduced Survey K Means Clustering')
        
       
        dbs=self.dbscan(scale_data)
        self.plot_graph(r_data,dbs.labels_,labelx='pc1',labely='pc2',title='Reduced Survey DBSCAN Clustering')
        
        
        opt=self.optics(r_data)
        self.plot_graph(r_data,opt.labels_,labelx='pc1',labely='pc2',title='Reduced Survey Optics Clustering')
        
        
        
        birc=self.birch(r_data)
        self.plot_graph(r_data,birc.labels_,labelx='pc1',labely='pc2',title='Reduced Survey Birch Clustering')
        
        spec=self.spectral_clustering(r_data)
        self.plot_graph(r_data,spec.labels_,labelx='pc1',labely='pc2',title='Reduced Survey Spectral Clustering')

        
test=algos()
#d=test.try_different_features_in_raw_data()
y,x=test.calculate_dependancy_matrix_preporcessed()

test.plot_graph_normal(y,labelx='Threshold',labely='Dependency Value',title='Threshold vs Dependency Value',x=x,flag=False)


