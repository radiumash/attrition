import streamlit as st
from PIL import Image

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

class LogisticRegressionModel:
  def __init__(self):
    self.model = LogisticRegression()

  def train(self, X_train,y_train):
    self.model.fit(X_train,y_train);

  def predict(self,X_test):
    return self.model.predict(X_test)

  def evaluateAccuracy(self,X_test,y_test):
    pred = self.predict(X_test)
    if pred is not None:
      accuracy = accuracy_score(y_test, pred)
      return (f'Accuracy for Logistic Regression: {accuracy * 100:.2f}%')
    else:
      return ('Model need improvement!')

class NaiveBayesModel:
  def __init__(self):
    self.model = GaussianNB()

  def train(self, X_train,y_train):
    self.model.fit(X_train,y_train);

  def predict(self,X_test):
    return self.model.predict(X_test)

  def evaluateAccuracy(self,X_test,y_test):
    pred = self.predict(X_test)
    if pred is not None:
      accuracy = accuracy_score(y_test, pred)
      return (f'Accuracy for Naive Bayes: {accuracy * 100:.2f}%')
    else:
      return('Model need improvement!')

class SVCModel:
  def __init__(self):
    self.model = SVC()

  def train(self, X_train,y_train):
    self.model.fit(X_train,y_train);

  def predict(self,X_test):
    return self.model.predict(X_test)

  def evaluateAccuracy(self,X_test,y_test):
    pred = self.predict(X_test)
    if pred is not None:
      accuracy = accuracy_score(y_test, pred)
      return (f'Accuracy for SVC: {accuracy * 100:.2f}%')
    else:
      return ('Model need improvement!')


def create_column(value):
    if value=='Less than $40K':
        return 'very low'
    elif value=='$40K - $60K':
        return 'low'
    elif value=='$60K - $80K':
        return 'medium'
    elif value=='$80K - $120K':
        return 'high'
    elif value=='$120K +':
        return 'very high'
    
st.set_page_config(page_title='Attrition Prediction',layout="wide")

# Sidebar content
st.sidebar.header("Project Support")
st.sidebar.subheader("Sample CSV Download")
st.sidebar.text("Please click below to\ndownload sample CSV file\nfor the attrition prediction!")

with open("data/CustomerChurn.csv", "rb") as file:
    btn = st.sidebar.download_button(
        label="Download CSV",
        data=file,
        file_name="CustomerChurn.csv",
        mime="text/csv"
        )

st.sidebar.text("\n")

# Main content
img = Image.open("images/header.png")
st.image(img,use_column_width=True)

st.title("Attrition Prediction")
st.write("Welcome to CA1 App of Machine Learning & Pattern Recognition!")

uploaded_file = st.file_uploader("Choose a CSV file for attrition prediction", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
   
    model = st.radio("Choose you classification model from the below list.",["LR","NB","SVC","ALL"], captions=["Logistic Regression", "Naive Bayes", "Support Vector Classifier (SVC)","ALL"])
    if model == 'LR':
        st.write('You selected Logistic Regression.')
    elif model == 'NB':
        st.write('You selected Naive Bayse.')
    elif model == 'SVC':
        st.write('You selected Support Vector Classifier.')
    elif model == 'ALL':
        st.write('You selected all the models.')   
    else:
        st.write("You didn\'t select anything.")

    if st.button('Generate Accuracy', type="primary"):
        data['Income_class']=data['Income_Category'].apply(create_column)

        OE = OneHotEncoder(handle_unknown='ignore')
        E=OE.fit_transform(data[['Gender','Education_Level','Marital_Status','Income_class','Card_Category']])
        encoder_df = pd.DataFrame(E.toarray())
        colnames = OE.get_feature_names_out()
        encoder_df.columns=colnames
        D3=encoder_df
        D3.head()        

        dataF = data.drop(['Attrition_Flag','Gender','Education_Level','Marital_Status','Income_Category','Income_class','Card_Category'],axis=1)
        dataF.head()

        X = dataF.join(D3)
        y = data['Attrition_Flag']
        X.head()        

        X_under=X[['Customer_Age','Dependent_count','Months_on_book','Total_Relationship_Count','Months_Inactive','Contacts_Count','Credit_Limit','Total_Revolving_Bal','Total_Trans_Amt','Total_Trans_Ct']]
        X.drop(['Customer_Age','Dependent_count','Months_on_book','Total_Relationship_Count','Months_Inactive','Contacts_Count','Credit_Limit','Total_Revolving_Bal','Total_Trans_Amt','Total_Trans_Ct'],axis=1,inplace=True)        

        #Apply Minmaxscaller
        scaled_X = scaler.fit_transform(X_under)
        names = X_under.columns
        scaled_mm = pd.DataFrame(scaled_X,columns=names)
        scaled_mm.head()
        X = scaled_mm.join(X)

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=3)

        if model == 'LR':
            modelLR = LogisticRegressionModel()
            modelLR.train(X_train, y_train)
            st.write(modelLR.evaluateAccuracy(X_test, y_test))
        elif model == 'NB':
            modelNB = NaiveBayesModel()
            modelNB.train(X_train, y_train)
            st.write(modelNB.evaluateAccuracy(X_test, y_test))
        elif model == 'SVC':
            modelSVC = SVCModel()
            modelSVC.train(X_train, y_train)
            st.write(modelSVC.evaluateAccuracy(X_test, y_test))
        elif model == 'ALL':
            modelLR = LogisticRegressionModel()
            modelNB = NaiveBayesModel()
            modelSVC = SVCModel()

            # Train the models
            modelLR.train(X_train, y_train)
            modelNB.train(X_train, y_train)
            modelSVC.train(X_train, y_train)

            # Evaluate the accuracy of each model
            st.write(modelLR.evaluateAccuracy(X_test, y_test))
            st.write(modelNB.evaluateAccuracy(X_test, y_test))
            st.write(modelSVC.evaluateAccuracy(X_test, y_test))
        else:
            st.write("You didn\'t select anything.")


    