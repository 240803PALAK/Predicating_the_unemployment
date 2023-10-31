# Predicating_the_unemployment
Analysis and social impact of unemployment rate during Pandemics done using machine learning techniques.

    
    import pandas as pd
    data=pd.read_csv('main1.csv')
    data1=data.copy()
    import warnings
    warnings.filterwarnings('ignore')
    data1.isnull().sum()
    data1['Gender'].unique()
    data1['Gender']=data1['Gender'].map({'Male':1,'Female':0})
    data1['Qualification'].unique()
    data1['Qualification']=data1['Qualification'].map({'Bachelor Degree':5,'Diploma':4,'PUC':3 ,'BCA':2 ,'PG':1 ,'Phd':0})
    data1['Origin'].unique()
    data1['Origin']=data1['Origin'].map({'Urban':3,'Remote':2,'Rural':1,'Semi urban':0})
    data1['Company_type'].unique()
    data1['Company_type']=data1['Company_type'].map({'Private limited company':6,'Corporative':5,'Partnership':4,'International company':3,
                                     'Not yet working':2, 'Nonprofit Organization':1,'Private institution':0})     
    data1['Status'].unique()
    data1['Status']=data1['Status'].map({'Employed':1,'Unemployed':0})
    x=data1.drop('Status',axis=1)
    y=data1['Status']
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
    from sklearn.linear_model import LogisticRegression
    lr=LogisticRegression()
    lr.fit(x_train,y_train)
    y_pred1=lr.predict(x_test)
    from sklearn.metrics import accuracy_score
    score1=accuracy_score(y_test,y_pred1)
    score1
    new_data=pd.DataFrame({
        'Gender':1,
        'Qualification':5,
        'Passout':2007,
        'Origin':0,
        'Experience':5,
        'Company_type':4,
    },index=[0])
    lr=LogisticRegression()
    lr.fit(x,y)
    p=lr.predict(new_data)
    prob=lr.predict_proba(new_data)
    if p==1:
        print("Employed")
        print(f"You will be employed with probability of {prob[0][1]:.2f}")
    else:
        print("Unemployed")
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    sns.countplot(x='Experience',hue='Status',data=data1)
    plt.subplot(1,2,2)
    sns.countplot(x='Passout',hue='Status',data=data1)
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    sns.countplot(x='Gender',hue='Status',data=data1)
    plt.subplot(1,2,2)
    sns.countplot(x='Origin',hue='Status',data=data1)
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    sns.countplot(x='Qualification',hue='Status',data=data1)
    plt.subplot(1,2,2)
    sns.countplot(x='Company_type',hue='Status',data=data1)
