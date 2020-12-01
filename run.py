# -*- coding: utf-8 -*-
"""
Reproducing the output scipt, for ML project 1.
@author: Lamiae Omarie Alaoui, Nessreddine LOUDIY
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *

def main():
    """Load the training data into feature matrix, class labels, and event ids:"""
    DATA_TRAIN_PATH ="../data/train.csv"
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH,sub_sample=False)
    
    def deal_with_missing_values(tX):
        for i in range(tX.shape[1]):
            
            a=np.where(tX[:,i]==-999)[0]
            b=np.where(tX[:,i]!=-999)
            
            if len(a)>0:
                tX_withoutmissingvalues=tX[:,i][b]
                median=np.median(tX_withoutmissingvalues)
                tX[:,i][a]=median
        return tX
    
    def deal_with_outliers(tX):
        for i in range(tX.shape[1]):
            q1=np.percentile(tX[:,i],25)
            q3=np.percentile(tX[:,i],75)
            IQR=q3-q1
            borne_inf=q1-1.5*IQR
            borne_sup=q3+1.5*IQR
            a=np.where(tX[:,i]<borne_inf)
            b=np.where(tX[:,i]>borne_sup)
            
            if len(a[0])>0:
                tX[:,i][a]=q1
            if len(b[0])>0:
                tX[:,i][b]=q3
        return tX
    
    def delete_features_with_low_variance(tX,threshold):
        a=np.std(tX,axis=0)
        b=np.where(a<threshold)
        for i in range (len(b)):
            tX=np.delete(tX,b[i],1)
        return b,tX
    def pearson_correlation(tX):
        pearson=np.zeros((tX.shape[1],tX.shape[1]))
        for i in range(tX.shape[1]):
            for j in range(i,tX.shape[1]):
                pearson[i,j]=np.corrcoef(tX[:,i], tX[:,j])[0,1]
                pearson[j,i]=pearson[i,j]
        return pearson  

    def transform_y(y):
        yb = np.ones(len(y))
        yb[np.where(y==-1)] = 0
        return yb.reshape((len(y),1))
    
    yb=transform_y(y)
    tX=deal_with_missing_values(tX)
    pearson=pearson_correlation(tX)
    #delete feature thatare highly correlated with an other feature
    c=0
    tX=np.delete(tX,9-c,1)
    c+=1
    tX=np.delete(tX,29-c,1)
    tX=deal_with_outliers(tX)
    b,tX=delete_features_with_low_variance(tX,0.0001)
    tX=standardize(tX)
    
    """least squares"""
    tx = np.c_[np.ones((y.shape[0], 1)), tX]
    w,loss=least_squares(y, tx)
    
    """Logistic regression"""
    gamma=1.2e-6
    
    max_iters =5000
    tx = np.c_[np.ones((y.shape[0], 1)), tX]
    
    initial_w=w
    initial_w=initial_w.reshape((tx.shape[1],1))
    w,loss=logistic_regression(yb, tx, initial_w,max_iters,gamma)
    
    DATA_TEST_PATH = "../data/test.csv"
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    
    
    tX_test=deal_with_missing_values(tX_test)
    c=0
    tX_test=np.delete(tX_test,9-c,1)
    c+=1
    tX_test=np.delete(tX_test,29-c,1)
    tX_test=deal_with_outliers(tX_test)
    for i in range (len(b)):
        tX_test=np.delete(tX_test,b[i],1)
    tX_test=standardize(tX_test)
    tx_test = np.c_[np.ones((tX_test.shape[0], 1)), tX_test]
    OUTPUT_PATH = '../data/output_Lamiae_Nessreddine.csv'
    y_pred = predict_labels(w, tx_test)
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
    
if __name__ == "__main__":
    main()
