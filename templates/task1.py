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
data = pd.read_csv("C:\Users\Aishwarya R\Desktop\flaskdemo\train_AV3.csv")
data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
data['Married'] = data['Married'].fillna(data['Married'].mode()[0])
data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])
data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
# using median values to impute the numerical columns
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].median())
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].median())
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].median())
data = data[data['ApplicantIncome'] < 25000]
data = data[data['CoapplicantIncome'] < 10000]
data = data[data['LoanAmount'] < 400]
#data['ApplicantIncome'] = np.log(data['ApplicantIncome'])
#data['CoapplicantIncome'] = np.log1p(data['CoapplicantIncome'])
data = data.drop(['Loan_ID'], axis = 1)
data['Gender'] = data['Gender'].replace(('Male','Female'),(1, 0))
data['Married'] = data['Married'].replace(('Yes','No'),(1, 0))
data['Education'] = data['Education'].replace(('Graduate','Not Graduate'), (1, 0))
data['Self_Employed'] = data['Self_Employed'].replace(('Yes','No'), (1, 0))
data['Loan_Status'] = data['Loan_Status'].replace(('Y','N'), (1, 0))
# as seen above that Urban and Semi Urban Property have very similar Impact on Loan Status, so, we will merge them together
data['Property_Area'] = data['Property_Area'].replace(('Urban','Semiurban', 'Rural'),(2, 1, 0))
# as seen above that apart from 0 dependents, all are similar hence, we merge them to avoid any confusion
data['Dependents'] = data['Dependents'].replace(('0', '1', '2', '3+'), (0, 1, 2, 3))
y = data['Loan_Status']
x = data.drop(['Loan_Status'], axis = 1)
x_resample, y_resample  = SMOTE().fit_sample(x, y.values.ravel())
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_resample, y_resample, test_size = 0.2, random_state = 0)
model =  GradientBoostingClassifier()
model.fit(x_train, y_train)
file = open('loan_status.pkl', 'wb')
pickle.dump(model, file)