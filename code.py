#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 15:31:42 2021

@author: sashank
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


data=pd.read_csv("data.csv")
data.material=[1 if(x=="abs") else 0 for x in data.material]
data.infill_pattern=[1 if(x=="grid") else 0 for x in data.infill_pattern]


x=data.drop('strength',axis=1)
y=data.strength
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

LR=LinearRegression()
LR.fit(x_train,y_train)

y_pred=LR.predict(x_test)
score=r2_score(y_test,y_pred)
print("R2 score is {0}".format(score))
lr_coef=list(LR.coef_)
lr_intc=LR.intercept_