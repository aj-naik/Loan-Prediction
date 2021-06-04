import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['Gender'] = train['Gender'].map({'Male':0,'Female':1})
train['Married'] = train['Married'].map({'No':0, 'Yes':1})
train['Loan_Status'] = train['Loan_Status'].map({'N':0,'Y':1})

train = train.dropna()
test = test.dropna()

X = train[['Gender','Married','ApplicantIncome','LoanAmount','Credit_History']]
y = train.Loan_Status
X.shape, y.shape

x_train , x_test , y_train, y_test = train_test_split(X,y,test_size=0.2,random_state= 10)

model = RandomForestClassifier(max_depth = 4, random_state = 10)
model.fit(x_train,y_train)

pred_test = model.predict(x_test)
s1=accuracy_score(y_test,pred_test)
print(s1)

pred_train = model.predict(x_train)
s2=accuracy_score(y_train,pred_train)
print(s2)

weights =open("classifier.pkl",mode = "wb")
pickle.dump(model,weights)
weights.close()
