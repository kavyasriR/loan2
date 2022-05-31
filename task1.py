import pandas as pd
import pickle
import imblearn
import numpy as np
from imblearn.over_sampling import SMOTE
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
data = pd.read_csv("train_AV3.csv")
data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
data['Married'] = data['Married'].fillna(data['Married'].mode()[0])
data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])
data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].median())
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].median())
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].median())
data = data[data['ApplicantIncome'] < 25000]
data = data[data['CoapplicantIncome'] < 10000]
data = data[data['LoanAmount']<10000000]
data = data.drop(['Loan_ID'], axis = 1)
data['Gender'] = data['Gender'].replace(('Male','Female'),(1, 0))
data['Married'] = data['Married'].replace(('Yes','No'),(1, 0))
data['Education'] = data['Education'].replace(('Graduate','Not Graduate'), (1, 0))
data['Self_Employed'] = data['Self_Employed'].replace(('Yes','No'), (1, 0))
data['Loan_Status'] = data['Loan_Status'].replace(('Y','N'), (1, 0))
data['Property_Area'] = data['Property_Area'].replace(('Urban','Semiurban', 'Rural'),(2, 1, 0))
data['Dependents'] = data['Dependents'].replace(('0', '1', '2', '3+'), (0, 1, 2, 3))
y = data['Loan_Status']
x = data.drop(['Loan_Status'], axis = 1)
oversample = SMOTE()
x, y = oversample.fit_resample(x, y)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
model =  GradientBoostingClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(accuracy_score(y_pred,y_test))
file = open('saved_model.pkl', 'wb')
pickle.dump(model, file)