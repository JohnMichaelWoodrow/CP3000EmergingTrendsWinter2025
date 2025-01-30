
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc, f1_score
import pickle
#-------------------------------------------------------------------------------------------------------------------------------
pd.set_option('display.max_columns', None )
import warnings                                                     # Importing warning to disable runtime warnings
warnings.filterwarnings("ignore")
#Change the file Path as per your needs
data = pd.read_csv('C:/Users/rusty/PycharmProjects/CP3000EmergingTrendsWinter2025/CP3000EmergingTrendsWinter2025/creditcard_2023.csv')


all(data.index == data.id)

data.set_index('id', inplace = True)


data['Class'].value_counts()

# Lets check for duplicates if any
data.duplicated().any()
## Dropping the duplicated values
data.drop_duplicates(keep = 'first', inplace = True)
data.duplicated().any()

def generate_corr_mat(df, high = 0.6, low = 0.1):
    """
        Filter the parameters based on their correlations' magnitude
    """
    corr_mat = {'highly_corr': dict(), 'low_corr': dict()}
    df = df.abs()
    cols = df.columns.tolist()
    for i, val in df.iterrows():
        corr_mat['highly_corr'][i] = [col for col in cols if val[col] > high and col != i]
        corr_mat['low_corr'][i] = [col for col in cols if val[col] < low]
        
    return corr_mat
corr_df = data.corr(method = 'spearman').round(2)
mask = np.triu(np.ones_like(corr_df, dtype=bool))
correlation_matrix = generate_corr_mat(corr_df)

x = data.drop(columns = ['Class'], axis=1)
y = data.Class

sc = StandardScaler()


x_scaled = sc.fit_transform(x) 
# pickle.dumps(open("scaler.pkl", 'wb'), sc)
x_scaled_df = pd.DataFrame(x_scaled,columns=x.columns)
x_scaled_df.head()

pca = PCA(n_components = 12)
x_new = pd.DataFrame(pca.fit_transform(x_scaled), columns = ['Col_'+ str(i) for i in range(12)])
# pickle.dumps(open('pca.pkl', 'wb'), pca)
x_new.head(3)

x_train,x_test,y_train,y_test = train_test_split(x_new, y, test_size=0.25, random_state=15, stratify= y)
cv = StratifiedKFold(n_splits = 8, shuffle = True)

def train_model(model, X_train, y_train, X_test, y_test):
    # Fit model
    model.fit(X_train, y_train)
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred)
    ## ROC AUC
    prob = model.predict_proba(X_test)  
    prob = prob[:, 1]
    fper, tper, _ = roc_curve(y_test, prob)
    auc_scr = auc(fper, tper)
    # Return evaluation metrics
    return model, accuracy, f1, roc_auc

model_lr = LogisticRegression()
time_start = time.time()
model_lr, acc_lr, f1_lr, roc_auc_lr = train_model(model_lr, x_train, y_train, x_test, y_test)
time_taken_lr = time.time() - time_start

model_gnb = GaussianNB()
time_start = time.time()
model_gnb, acc_gnb, f1_gnb, roc_auc_gnb = train_model(model_gnb, x_train, y_train, x_test, y_test)
time_taken_gnb = time.time() - time_start

model_dt = DecisionTreeClassifier()
time_start = time.time()
model_dt, acc_dt, f1_dt, roc_auc_dt = train_model(model_dt, x_train, y_train, x_test, y_test)
time_taken_dt = time.time() - time_start

model_rf = RandomForestClassifier()
time_start = time.time()
model_rf, acc_rf, f1_rf, roc_auc_rf = train_model(model_rf, x_train, y_train, x_test, y_test)
time_taken_rf = time.time() - time_start

model_xgb = xgb.XGBRFClassifier()
time_start = time.time()
model_xgb, acc_xgb, f1_xgb, roc_auc_xgb = train_model(model_xgb, x_train, y_train, x_test, y_test)
time_taken_xgb = time.time() - time_start

accuracies = [acc_lr, acc_gnb, acc_dt, acc_rf, acc_xgb]
f_score = [f1_lr, f1_gnb, f1_dt, f1_rf, f1_xgb]
roc_auc = [roc_auc_lr, roc_auc_gnb, roc_auc_dt, roc_auc_rf, roc_auc_xgb]
time = [time_taken_lr, time_taken_gnb, time_taken_dt, time_taken_rf, time_taken_xgb]

final_df = pd.DataFrame({"Accuracies": accuracies, "F1 Scores": f_score, "ROC AUC": roc_auc, "Time Taken": time}, 
                       index = ['LogisticReg', 'GaussianNB', 'DecisionTree', 'RandomForest', 'XGB'])
final_df = final_df.round(4)
final_df

# # Finding the model with the maximum ROC AUC Score
# max_roc_auc_index = final_df['ROC AUC'].idxmax()
# best_model_name = max_roc_auc_index
# best_model_roc_auc_score = final_df.loc[max_roc_auc_index, 'ROC AUC']

# # Printing the best model name and its ROC AUC score
# print(f"BestModel:{best_model_name}")
# print(f"ROCAUC:{best_model_roc_auc_score}")

# Finding the model with the highest F1 score
max_f1_index = final_df['F1 Scores'].idxmax()
best_model_name = max_f1_index
best_model_f1_score = final_df.loc[max_f1_index, 'F1 Scores']

# Printing the best model name and its F1 score
print(f"Number of records: {data.shape[0]}")
print(f"Number of features: {data.shape[1]}\n")
print(f"BestModel:{best_model_name}")
print(f"F1Score:{best_model_f1_score}")


