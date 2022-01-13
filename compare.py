import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
import numpy as np



#y = ""
X = ""
#clf = " "
np.random.seed(70)
def compare():
    

    st.header("""
         Explore five different classifiers
         Below is the credit score german dataset obtained from UCI?
         """ )

data = st.file_uploader("Upload dataset:",type=['csv','xlsx'])
st.success("Data successfully loaded")

if data is not None:
    df=pd.read_csv(data,';')
    st.dataframe(df)

    le_CreditHistory = LabelEncoder()
    df['CreditHistory'] = le_CreditHistory.fit_transform(df['CreditHistory'])
    df["CreditHistory"].unique()

    le_Employment = LabelEncoder()
    df['Employment'] = le_Employment.fit_transform(df['Employment'])
    df["Employment"].unique()



ok = st.sidebar.checkbox('Select Multiple Columns')
if ok:
    new_data = st.multiselect('Select preferred colunmn features',df.columns)
    df1=df[new_data]
    st.dataframe(df1)
   
    X = df1.iloc[:,0:-1]
    y = df1.iloc[:,-1]
    
   
    seed=st.sidebar.slider('Seed',1,200)

    classifier_name = st.sidebar.selectbox('Select the Classifier:',('SVM','LogisticRegression','Decision tree','RandomForest','XgBoost'))

    def add_parameter(name_of_clf):
        param = dict()
        if name_of_clf == 'SVM':
            C=st.sidebar.slider('C',1,15)
            param['C']=C
        
        else:
              name_of_clf=='RandonForest'
              n_estimators =st.sidebar.slider('n_estimators',int(70))
              param['n_estimators']=n_estimators
        return param
            
    #call the function
    param =add_parameter(classifier_name)
    
    def get_classifier(name_of_clf,param):
        
        if name_of_clf == 'SVM':
            clf=SVC(C=param['C'])
        elif name_of_clf == 'LogisticRegression':
            clf=LogisticRegression()
        elif name_of_clf == 'Decision tree':
            clf=DecisionTreeClassifier()
        elif name_of_clf == 'RandomForest':
            clf=RandomForestClassifier()
        elif name_of_clf == 'XgBoost':
            clf = XGBClassifier()
        else:
            st.warning('Select Algorithm')
        return clf
        
    clf=get_classifier(classifier_name,param)
       
if data is not None:  
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=seed,shuffle=True)

    clf.fit(X_train,y_train)

    y_pred=clf.predict(X_test)

#st.write('Predictions:',y_pred)
    accuracy=accuracy_score(y_test,y_pred)
    st.write('Name of classifier:',classifier_name)
    st.write('Accuracy',accuracy)
    st.write("Classifier report:",classification_report(y_test, y_pred))
    st.write("Confusion_Matrix",confusion_matrix(y_test,y_pred))





