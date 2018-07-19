# Importing the libraries
import pandas as pd
import numpy as np
from  sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

#Read the data
data=pd .read_csv("appendix.csv")
data .head()
print (data.head())

# Removing the unwanted rows and columns and reading the required data
data=data.iloc[:,[0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,22]]
data.head()
print(data.head())
print(data)

# Numbering same data using LabelEncoder (Institute and subjects)
enc=LabelEncoder()
data.iloc[:,0]=enc.fit_transform(data.iloc[:,0])
data.head()
print(data)
data.iloc[:,4]=enc.fit_transform(data.iloc[:,4])
data.head()
print(data)

# Considering the independent variables and label together as X which is the input
X=data.iloc[:,[0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18]]
X.head()
print("X")

# We have to predict which is the best Institution and it is the target variable represented as  y
y=data.Institution

# Trying to create dummies 
X=pd.get_dummies(X,drop_first=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

# Fitting the regression model
linear=LinearRegression()
linear.fit(X_train,y_train)
print(linear.fit(X_train,y_train))
# Predicting with the created model
pred=linear.predict(X_test)
# Cheking the effiency of the model
r2_score(y_test,pred)
print(r2_score(y_test,pred))
