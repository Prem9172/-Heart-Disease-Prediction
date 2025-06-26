#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas numpy scikit-learn streamlit


# In[2]:


pip install flask


# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np


# In[4]:


# Load dataset
data = pd.read_csv(r"C:\Users\premt\Downloads\heart.csv")


# In[5]:


data.head()


# In[6]:


data.isnull().sum()


# In[7]:


data_dup = data.duplicated().any()


# In[8]:


data_dup


# In[9]:


data = data.drop_duplicates()


# In[10]:


data_dup = data.duplicated().any()


# In[11]:


# data processing
cate_val=[]
cont_val=[]

for column in data.columns: 
    if data[column].nunique() <=10:
        cate_val.append(column)
    else:
        cont_val.append(column)


# In[12]:


cate_val


# In[13]:


cont_val


# In[14]:


# encoding  categorical Data 


cate_val


# In[15]:


cate_val.remove('sex')
cate_val.remove('target')
data = pd.get_dummies(data,columns=cate_val,drop_first=True)


# In[16]:


data.head()


# In[17]:


data.head()


# In[18]:


from sklearn.preprocessing import StandardScaler


# In[19]:


st = StandardScaler()
data[cont_val]=st.fit_transform(data[cont_val])


# In[20]:


data.head()


# In[96]:


# Splitting The data set into the traning set and test set 
data.drop(['exang_1','fbs_1','restecg_1','restecg_2','slope_1','slope_2','thal_1','thal_2','thal_3' ,'exang_1','fbs_1','restecg_1','restecg_2','slope_1'],axis=1)


# In[97]:


X = data.drop('target',axis=1).values


# In[98]:


X


# In[99]:


y = data['target'].values


# In[100]:


y


# In[101]:


from sklearn.model_selection import train_test_split


# In[102]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)


# In[103]:


X_train


# In[104]:


y_train


# In[105]:


X_test


# In[106]:


y_test


# In[107]:


data.head()


# In[108]:


from sklearn.linear_model import LogisticRegression


# In[109]:


log=LogisticRegression()
log.fit(X_train,y_train)


# In[110]:


y_predl = log.predict(X_test)


# In[111]:


from sklearn.metrics import accuracy_score


# In[112]:


accuracy_score(y_test,y_predl)


# In[113]:


from sklearn import svm


# In[114]:


svm = svm.SVC()


# In[115]:


svm.fit(X_train,y_train)


# In[116]:


y_pred2 = svm.predict(X_test)


# In[117]:


accuracy_score(y_test,y_pred2)


# In[118]:


# kNeighborsClassifier()


# In[119]:


from sklearn.neighbors import KNeighborsClassifier


# In[120]:


KNN = KNeighborsClassifier()


# In[121]:


KNN.fit(X_train,y_train)


# In[122]:


y_pred3 = KNN.predict(X_test)


# In[123]:


accuracy_score(y_test,y_pred3)


# In[124]:


score =[]

for k in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    score.append(accuracy_score(y_test, y_pred))


# In[67]:


score


# In[125]:


#non-liner ml algorithm 

data.head()


# In[127]:


data=data.drop_duplicates()


# In[128]:


data.shape


# In[129]:


X=data.drop('target',axis=1)
y=data['target']


# In[130]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)


# In[131]:


# decison tree classifer

from sklearn.tree import DecisionTreeClassifier


# In[132]:


dt = DecisionTreeClassifier()


# In[133]:


dt.fit(X_train,y_train)


# In[134]:


y_pred4 = dt.predict(X_test)


# In[135]:


accuracy_score(y_test, y_pred4)


# In[136]:


from sklearn.ensemble import RandomForestClassifier


# In[137]:


rf = RandomForestClassifier()


# In[138]:


rf.fit(X_train,y_train)


# In[139]:


y_pred5 = rf.predict(X_test)


# In[140]:


accuracy_score(y_test, y_pred5)


# In[141]:


from sklearn.ensemble import GradientBoostingClassifier


# In[142]:


gbc=GradientBoostingClassifier()


# In[143]:


gbc.fit(X_train,y_train)


# In[144]:


y_pred6 = gbc.predict(X_test)


# In[145]:


accuracy_score(y_test, y_pred6)


# In[146]:


final=pd.DataFrame({'Model':['LR','SVM','DT','RF','GB'],'ACC':[accuracy_score(y_test,y_predl),
                                                              accuracy_score(y_test,y_pred2),
                                                              accuracy_score(y_test,y_pred4),
                                                              accuracy_score(y_test,y_pred5),
                                                              accuracy_score(y_test,y_pred6)]
                })


# In[147]:


final


# In[148]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[149]:


sns.barplot(x='Model', y='ACC', data=final)

plt.xticks(rotation=45)  # Optional: rotate x-labels for better readability
plt.title("Model vs Accuracy")
plt.show()


# In[150]:


import pandas as pd 


# In[160]:


data.head()


# In[161]:


data.info()


# In[162]:


new_data = pd.DataFrame({
    'age':52,
    'sex':1,
    'trestbps':0,
    'chol':212,
    'thalach':2,
    'oldpeak':0,
    'cp_1':0,
    'cp_2':0,
    'cp_3':0,
    'fbs_1':1,
    'restecg_1':1,
    'restecg_2':1,
    'exang_1':2,
    'slope_1':0,
    'slope_2':0,
    'ca_1':0,
    'ca_2':0,
    'ca_3':0,
    'ca_4':0,
    'thal_1':1,
    'thal_2':3,
    'thal_3':4
    },index=[0])


# In[163]:


new_data.values


# In[164]:


p= rf.predict(new_data)


# In[165]:


if p[0] == 0:
    print("No Disease")
else:
    print("Disease")


# In[ ]:


from tkinter import *
import joblib

def show_entry_fields():
    p1 = int(e1.get())
    p2 = int(e2.get())
    p3 = int(e3.get())
    p4 = int(e4.get())
    p5 = int(e5.get())
    p6 = int(e6.get())
    p7 = int(e7.get())
    p8 = int(e8.get())
    p9 = int(e9.get())
    p10 = float(e10.get())
    p11 = int(e11.get())
    p12 = int(e12.get())
    p13 = int(e13.get())

    model = joblib.load('model_joblib_heart')
    result = model.predict([[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13]])
            
            
    if result[0] == 0:
        result_label.config(text="No Heart Disease", fg="green")
    else:
        result_label.config(text="Heart Disease Detected", fg="red")

 # Main Window
master = Tk()
master.title("Heart Disease Predictor System")
Label(master, text="Heart Disease Prediction System", bg="black", fg="white", font=('Arial', 14)).grid(row=0, columnspan=2, pady=10)

 # Input Labels
Label(master, text="Enter Your Age").grid(row=1)
Label(master, text="Sex [1=Male, 0=Female]").grid(row=2)
Label(master, text="Chest Pain Type (cp)").grid(row=3)
Label(master, text="Resting Blood Pressure (trestbps)").grid(row=4)
Label(master, text="Cholesterol (chol)").grid(row=5)
Label(master, text="Fasting Blood Sugar > 120? [1/0]").grid(row=6)
Label(master, text="Resting ECG Results (restecg)").grid(row=7)
Label(master, text="Maximum Heart Rate (thalach)").grid(row=8)
Label(master, text="Exercise Induced Angina (exang)").grid(row=9)
Label(master, text="Oldpeak").grid(row=10)
Label(master, text="Slope").grid(row=11)
Label(master, text="CA (Number of Major Vessels)").grid(row=12)
Label(master, text="Thal (0=Normal, 1=Fixed Defect, 2=Reversible Defect)").grid(row=13)

 # Entry Fields
e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)
e9 = Entry(master)
e10 = Entry(master)
e11 = Entry(master)
e12 = Entry(master)
e13 = Entry(master)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)
e9.grid(row=9, column=1)
e10.grid(row=10, column=1)
e11.grid(row=11, column=1)
e12.grid(row=12, column=1)
e13.grid(row=13, column=1)

 # Predict Button
Button(master, text='Predict', command=show_entry_fields).grid(row=14, columnspan=2, pady=10)

 # Result Label
result_label = Label(master, text="", font=('Arial', 12))
result_label.grid(row=15, columnspan=2, pady=10)

master.mainloop()


# In[ ]:





# In[ ]:




