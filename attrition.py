import streamlit as st
from PIL import Image

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score,precision_score
from sklearn.model_selection import cross_val_score

class LogisticRegressionModel:
  def __init__(self):
    self.model = LogisticRegression()

  def train(self, X_train,y_train):
    self.model.fit(X_train,y_train);

  def predict(self,X_test):
    return self.model.predict(X_test)

  def crossValScore(self,X,y,cv):
    return cross_val_score(LogisticRegression(),X,y,cv)

  def evaluateAccuracy(self,X_test,y_test):
    pred = self.predict(X_test)
    if pred is not None:
      accuracy = accuracy_score(y_test, pred)
      precision = precision_score(y_test, pred, average="binary", pos_label="Existing Customer")
      recall = recall_score(y_test, pred, average="binary", pos_label="Existing Customer")
      result = "Logistic Regression ________________________________________________ | "
      result += f'Accuracy: {accuracy * 100:.2f}% | '
      result += f'Precision: {precision:.4f} | '
      result += f'Recall: {recall:.4f}'
      return result

    else:
      return 'Model need improvement!'

class NaiveBayesModel:
  def __init__(self):
    self.model = GaussianNB()

  def train(self, X_train,y_train):
    self.model.fit(X_train,y_train);

  def predict(self,X_test):
    return self.model.predict(X_test)

  def crossValScore(self,X,y,cv):
    return cross_val_score(GaussianNB(),X,y,cv)

  def evaluateAccuracy(self,X_test,y_test):
    pred = self.predict(X_test)
    if pred is not None:
      accuracy = accuracy_score(y_test, pred)
      precision = precision_score(y_test, pred, average="binary", pos_label="Existing Customer")
      recall = recall_score(y_test, pred, average="binary", pos_label="Existing Customer")
      result = "Naive Bayse ________________________________________________ | "
      result += f'Accuracy: {accuracy * 100:.2f}% | '
      result += f'Precision: {precision:.4f} | '
      result += f'Recall: {recall:.4f}'
      return result
    else:
      print('Model need improvement!')

class SVCModel:
  def __init__(self):
    self.model = SVC()

  def train(self, X_train,y_train):
    self.model.fit(X_train,y_train);

  def predict(self,X_test):
    return self.model.predict(X_test)

  def crossValScore(self,X,y,cv):
    return cross_val_score(SVC(),X,y,cv)

  def evaluateAccuracy(self,X_test,y_test):
    #st.inpu()
    pred = self.predict(X_test)
    if pred is not None:
      accuracy = accuracy_score(y_test, pred)
      precision = precision_score(y_test, pred, average="binary", pos_label="Existing Customer")
      recall = recall_score(y_test, pred, average="binary", pos_label="Existing Customer")
      result = "Support Vector Classifier ________________________________________________ | "
      result += f'Accuracy: {accuracy * 100:.2f}% | '
      result += f'Precision: {precision:.4f} | '
      result += f'Recall: {recall:.4f}'
      return result
    else:
      print('Model need improvement!')


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


# Get interger array for the category values
def GetOneHotEncoder(ddf):
  Encoder = OneHotEncoder(handle_unknown='ignore')
  E = Encoder.fit_transform(ddf[ddf.columns])
  encoder_df = pd.DataFrame(E.toarray())
  colnames = Encoder.get_feature_names_out()
  encoder_df.columns=colnames
  return encoder_df

def preProcessData(data):
 
  # Numerical Features
  data['Customer_Age'].fillna(int(data['Customer_Age'].mean()),inplace=True)
  data['Dependent_count'].fillna(int(data['Dependent_count'].mean()),inplace=True)
  data['Months_on_book'].fillna(int(data['Months_on_book'].mean()),inplace=True)
  data['Total_Relationship_Count'].fillna(int(data['Total_Relationship_Count'].mean()),inplace=True)
  data['Months_Inactive'].fillna(int(data['Months_Inactive'].mean()),inplace=True)
  data['Contacts_Count'].fillna(int(data['Contacts_Count'].mean()),inplace=True)
  data['Credit_Limit'].fillna(int(data['Credit_Limit'].mean()),inplace=True)
  data['Total_Revolving_Bal'].fillna(int(data['Total_Revolving_Bal'].mean()),inplace=True)
  data['Total_Trans_Ct'].fillna(int(data['Total_Trans_Ct'].mean()),inplace=True)

  # Categorical Features
  data['Gender'].fillna(data['Gender'].mode().iloc[0],inplace=True)
  data['Education_Level'].fillna(data['Education_Level'].mode().iloc[0],inplace=True)
  data['Marital_Status'].fillna(data['Marital_Status'].mode().iloc[0],inplace=True)
  data['Income_Category'].fillna(data['Income_Category'].mode().iloc[0],inplace=True)
  data['Card_Category'].fillna(data['Card_Category'].mode().iloc[0],inplace=True)
  data['Attrition_Flag'].fillna(data['Attrition_Flag'].mode().iloc[0],inplace=True)

  data['Income_class'] = data['Income_Category'].apply(create_column)
  data.drop(['Income_Category'],axis=1,inplace=True)

  # 
  X_scale = data[['Credit_Limit','Total_Trans_Amt']]
  header = X_scale.columns
  normalizer = StandardScaler()
  N = normalizer.fit_transform(X_scale)
  df = pd.DataFrame(N)

  df.columns = header
  X_normal = data.drop(['Credit_Limit','Total_Trans_Amt'], axis=1)
  data = X_normal.join(df)

  OHE = data[['Gender','Education_Level','Marital_Status','Income_class','Card_Category']]

  AOHE = GetOneHotEncoder(OHE)
  ndata = data.drop(['Gender','Education_Level','Marital_Status','Income_class','Card_Category'],axis=1)
  ndf = ndata.join(AOHE)
  return ndf

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
tab1, tab2 = st.tabs(["🗃 CSV Data", "📈 1 Entity"])

tab1.subheader("Train Model & get Accuracy")

img = Image.open("images/header.png")
tab1.image(img,use_column_width=True)

tab1.title("Classification of Customer Attrition using Machine Learning")
tab1.write("Welcome to CA1 App of Machine Learning & Pattern Recognition!")

uploaded_file = tab1.file_uploader("Choose a CSV file for attrition prediction", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
   
    model = tab1.radio("Choose your classification model from the below list.",["LR","NB","SVC","ALL"], captions=["Logistic Regression", "Naive Bayes", "Support Vector Classifier (SVC)","ALL"])
    if model == 'LR':
        tab1.write('You selected Logistic Regression.')
    elif model == 'NB':
        tab1.write('You selected Naive Bayse.')
    elif model == 'SVC':
        tab1.write('You selected Support Vector Classifier.')
    elif model == 'ALL':
        tab1.write('You selected all the models.')   
    else:
        tab1.write("You didn\'t select anything.")

    if tab1.button('Generate Accuracy', type="primary"):
        
        X = preProcessData(data)
        y = data['Attrition_Flag']  
        X.drop(['Attrition_Flag'], axis=1, inplace=True)

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=3)

        if model == 'LR':
            modelLR = LogisticRegressionModel()
            modelLR.train(X_train, y_train)
            tab1.write(modelLR.evaluateAccuracy(X_test, y_test))
        elif model == 'NB':
            modelNB = NaiveBayesModel()
            modelNB.train(X_train, y_train)
            tab1.write(modelNB.evaluateAccuracy(X_test, y_test))
        elif model == 'SVC':
            modelSVC = SVCModel()
            modelSVC.train(X_train, y_train)
            tab1.write(modelSVC.evaluateAccuracy(X_test, y_test))
        elif model == 'ALL':
            modelLR = LogisticRegressionModel()
            modelNB = NaiveBayesModel()
            modelSVC = SVCModel()

            # Train the models
            modelLR.train(X_train, y_train)
            modelNB.train(X_train, y_train)
            modelSVC.train(X_train, y_train)

            # Evaluate the accuracy of each model
            tab1.write(modelLR.evaluateAccuracy(X_test, y_test))
            tab1.write(modelNB.evaluateAccuracy(X_test, y_test))
            tab1.write(modelSVC.evaluateAccuracy(X_test, y_test))
        else:
            tab1.write("You didn\'t select anything.")




with tab2.subheader("Input individual data to predict Accuracy."):
  with st.form("my_form"):
    st.write("Fill all the values and hit 'Get Accuracy' button!")
    with open("data/CustomerChurn.csv", "rb") as file:
      data = pd.read_csv(file)  
      Gender = st.selectbox(
          'Customer Gender',
          (data['Gender'].unique()))
      Customer_Age = st.selectbox(
          'Customer Age',
          (range(18,100)))
      Card_Category = st.selectbox(
          'Card Category',
          (data['Card_Category'].unique()))
      Education_Level = st.selectbox(
          'Customer Education',
          (data['Education_Level'].unique()))  
      Marital_Status	 = st.selectbox(
          'Marital Status',
          (data['Marital_Status'].unique()))  
      Income_Category = st.selectbox(
          'Customer Income Category',
          (data['Income_Category'].unique()))  
      Dependent_count = st.selectbox(
          'Dependent Count',
          (range(10)))
      Months_on_book = st.selectbox(
          'Customer Since (in months)',
          (range(1,500)))
      Total_Relationship_Count = st.selectbox(
          'Total no. of Products Held by the Customer',
          (range(1,50)))
      Months_Inactive	 = st.selectbox(
          'Inactive Since (in months)',
          (range(1,50)))
      Contacts_Count = st.selectbox(
          'Number of Contact by Customer',
          (range(1,50)))
      Credit_Limit = st.text_input('Credit Limit on the Credit Card', '')
      Total_Revolving_Bal = st.text_input('Total Revolving Balance on the Credit Card', '')
      Total_Trans_Amt = st.text_input('Total Transaction Amount (Last 12 months)', '')
      Total_Trans_Ct = st.text_input('Total Transaction Count (Last 12 months)', '')

      # Every form must have a submit button.
      submitted = st.form_submit_button("Get Accuracy")
      if submitted:
        if not Credit_Limit:
          st.warning('Please input Credit Limit on the Credit Card.')
          st.stop()
        if not Total_Revolving_Bal:
          st.warning('Please input Total Revolving Balance on the Credit Card.')
          st.stop()
        if not Total_Trans_Amt:
          st.warning('Please input Total Transaction Amount (Last 12 months).')
          st.stop()
        if not Total_Trans_Ct:
          st.warning('Please input Total Transaction Count (Last 12 months).')
          st.stop()                              
        dat = [[0, int(Customer_Age), Gender, int(Dependent_count),
          Education_Level, Marital_Status, Income_Category, Card_Category,
          int(Months_on_book), int(Total_Relationship_Count), int(Months_Inactive),
          int(Contacts_Count), int(Credit_Limit), float(Total_Revolving_Bal),
          float(Total_Trans_Amt), float(Total_Trans_Ct)]]
        
        #df = pd.DataFrame(dat,columns=cols)
        data.loc[len(data.index)] = [0, int(Customer_Age), Gender, int(Dependent_count),
          Education_Level, Marital_Status, Income_Category, Card_Category,
          int(Months_on_book), int(Total_Relationship_Count), int(Months_Inactive),
          int(Contacts_Count), int(Credit_Limit), float(Total_Revolving_Bal),
          float(Total_Trans_Amt), float(Total_Trans_Ct)]
        
        X = preProcessData(data)
        y = data['Attrition_Flag']  
        X.drop(['Attrition_Flag'], axis=1, inplace=True)

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=3)

        modelLR = LogisticRegressionModel()
        modelNB = NaiveBayesModel()
        modelSVC = SVCModel()

        # Train the models
        modelLR.train(X_train, y_train)
        modelNB.train(X_train, y_train)
        modelSVC.train(X_train, y_train)       

        predLR = modelLR.predict(X.tail(1))
        predNB = modelNB.predict(X.tail(1))
        predSVC = modelSVC.predict(X.tail(1))

        st.write(f"Predicted by LR: {predLR}")
        st.write(f"Predicted by NB: {predNB}")
        st.write(f"Predicted by SVC: {predSVC}")




                

        
















    