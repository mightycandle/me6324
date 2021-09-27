import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

file=open("values.csv")
data=pd.read_csv("data.csv")
data.infill_pattern=[1 if i =="grid" else 0 for i in data.infill_pattern]
data.material=[1 if i =="abs" else 0 for i in data.material]

data.drop('infill_pattern',axis=1)
data.drop('material',axis=1)

x=data
x=data.drop('strength',axis=1)
x=x.drop('roughness',axis=1)

y=data.strength

score=0
cur=0
thresh=0.95
lr_coef1=[]
lr_intc1=0

for epoch in range(1000):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0,shuffle=True)
    
    LR=LinearRegression()
    LR.fit(x_train,y_train)
    
    y_pred=LR.predict(x_test)
    cur=r2_score(y_test,y_pred)
    if(score<cur):
        score=cur
        lr_coef1=list(LR.coef_)
        lr_intc1=LR.intercept_
print("Max R2 score is {0}".format(score))

grid=np.loadtxt(file,delimiter=",")
a=[]
for i in range(len(grid)):
    cur=lr_intc1
    for j in range(len(lr_coef1)):
        cur+=lr_coef1[j]*grid[i][j]
    a.append([cur-y_train[i],i+1])
a.sort()
print(a)
# for i in range(20):
#     print(grid[a[i][1]])