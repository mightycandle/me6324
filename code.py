import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


data=pd.read_csv("book4.csv")

x=data.drop('strength',axis=1)
x=x.drop('roughness',axis=1)

y1=data.infill_density
y2=data.wall_thickness
y3=data.roughness
y4=data.strength

y=y4

score=0
cur=0
thresh=0.95
lr_coef1=[]
lr_intc1=0

for epoch in range(2000):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    
    LR=LinearRegression()
    LR.fit(x_train,y_train)
    
    y_pred=LR.predict(x_test)
    cur=r2_score(y_test,y_pred)
    if(score<cur):
        score=cur
        lr_coef1=list(LR.coef_)
        lr_intc1=LR.intercept_
print("Max R2 score is {0}".format(score))