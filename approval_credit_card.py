# Importing libraries
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load the dataset
path = '/Users/snehabarve/Documents/MISC/creditCardApplication.csv'
df = pd.read_csv(path)

# print(df.describe(include="all"))
# print(df.columns)
# print(df.isnull().sum())
# print(df.shape)

df = df.drop('ID', axis=1)

# print("Column names: ", df.columns)
# print("df.shape: ", df.shape)


# df["Gender"] = pd.get_dummies(df["Gender"])
# df["MaritalSatus"] = pd.get_dummies(df["MaritalStatus"])

# Handling Categorical Data
# label_encoder = LabelEncoder() # Initialize LabelEncoder
# for col in df.columns:
#     if df[col].dtype == 'object':
#         df[col] = label_encoder.fit_transform(df[col])

df['Gender'] = df['Gender'].replace({'M': 1, 'F': 2})
df['MaritalStatus'] = df['MaritalStatus'].replace({'SINGLE': 1, 'MARRIED': 2, 'DIVORCED': 3, 'WIDOWED': 4})

# features (x) and target (y)
x = df.drop('CREDIT_SCORE', axis=1)
y = df['CREDIT_SCORE']

# x = pd.get_dummies(x, drop_first=True)

# Initialize LabelEncoder and OneHotEncoder
# label_encoder = LabelEncoder()
# one_hot_encoder = OneHotEncoder()

# Fit and transform the encoder on the categorical column
# x = one_hot_encoder.fit_transform(x[["Gender", "MaritalStatus"]])
print(x.columns)

# Feature Scaling
mas = MaxAbsScaler()
x_scaled = mas.fit_transform(x)


# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=123)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(x_scaled, y)
y_pred_dt = dt_model.predict(x_scaled)
accuracy_dt = round(accuracy_score(y, y_pred_dt) * 100, 2)
# print("Accuracy:", accuracy_dt)

# Random Forest model
rf_model = RandomForestClassifier(random_state=422)
rf_model.fit(x_scaled, y)
y_pred_rf = rf_model.predict(x_scaled)
accuracy_rf = round(accuracy_score(y_pred_rf, y) * 100, 2)
# print("Accuracy:", accuracy_rf)

# Logistic Regression
logreg_model = LogisticRegression(max_iter=1500)
logreg_model.fit(x_scaled, y)
y_pred_log = logreg_model.predict(x_scaled)
accuracy_logreg = round(accuracy_score(y_pred_log, y) * 100, 2)
# print("Accuracy:", accuracy_logreg)

# KNN (k-Nearest Neighbors)
knn_model = KNeighborsClassifier()
knn_model.fit(x_scaled, y)
y_pred_knn = knn_model.predict(x_scaled)
accuracy_knn = round(accuracy_score(y_pred_knn, y) * 100, 2)
# print("Accuracy:", accuracy_knn)

# Stochastic Gradient Descent
sgd_model = SGDClassifier()
sgd_model.fit(x_scaled, y)
y_pred_sgd = sgd_model.predict(x_scaled)
accuracy_sgd = round(accuracy_score(y_pred_sgd, y) * 100, 2)
# print("Accuracy:", accuracy_sgd)

# Gradient Boosting Classifier
gbc_model = GradientBoostingClassifier()
gbc_model.fit(x_scaled, y)
y_pred_gbc = gbc_model.predict(x_scaled)
accuracy_gbc = round(accuracy_score(y_pred_gbc, y) * 100, 2)
# print("Accuracy:", accuracy_gbc)

# Support Vector Machines
svc_model = SVC()
svc_model.fit(x_scaled, y)
y_pred_svm = svc_model.predict(x_scaled)
accuracy_svm = round(accuracy_score(y_pred_svm, y) * 100, 2)
# test_score = svc_model.score(x_scaled, y)
# print("Test score SVM: ", test_score)
# print("Accuracy SVM:", accuracy_svm)


model_list = pd.DataFrame({
    'Model Names': ['Decision Tree', 'Random Forest', 'Logistic Regression', 'KNN',
                    'Stochastic Gradient Descent', 'Gradient Boosting Classifier', 'Support Vector Machine'],
    'Accuracy Score': [accuracy_dt, accuracy_rf, accuracy_logreg, accuracy_knn,
                       accuracy_sgd, accuracy_gbc, accuracy_svm]})

sorted_list = model_list.sort_values(by='Accuracy Score', ascending=False)
print(sorted_list)

with open("rf.model", "wb") as f:
    pickle.dump(rf_model, f)


# d = [[1, 25, 2, 70000, 2, 3]]
#
# res = rf_model.predict(d)
# res2 = dt_model.predict(d)
# res3 = gbc_model.predict(d)
# print("Result RF: ", res)
# print("Result DT: ", res2)
# print("Result GBC: ", res3)


with open("gbc.model", "wb") as f:
    pickle.dump(gbc_model, f)

with open("dt.model", "wb") as f:
    pickle.dump(dt_model, f)


