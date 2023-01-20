#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras import models, layers

import pandas as pd
import numpy as np
import streamlit as st

#
# # In[5]:

st.set_page_config(layout="wide")
st.write(""" # Automatisering av låneprosessen for firmaer""")

st.sidebar.title("Vennligst fyll inn opplysningene dine her:")
#
df_tran = pd.read_csv('transformed_data.csv', delimiter=',', low_memory=False)
#
#
# # In[6]:
#
#
st.sidebar.header("Inndata verdier")

st.write('---')
"""
Velkommen kjære bankkunde! \n
Hvis du er interessert i å se om et firma kan betjene et lån, kan du prøve dette programmet her.
Det eneste du må gjøre er å fylle inn opplysningene dine, så gjør programmet resten for deg.
Husk at dette bare er estimatorer basert på data hentet fra tidligere søkere. \n
Håper dette gjør prosessen lettere.
"""
#
# # In[21]:
#
#
#
#
# # In[22]:
#
#
#
#
# # ### Splitting the data into train and test
#
# # In[23]:
#
#
training_data = df_tran.sample(frac=0.8, random_state=21)
testing_data = df_tran.drop(training_data.index)
#

#
#
# # In[24]:
#
#
# # Code for scikit-learn based model
#
X = training_data.drop(["Default"], axis=1)
y = training_data["Default"]


# dict_verdier = {"Ja": 1, "Nei": 0}  # Lager en verdi for videre bruk


#
def brukerverdier():
    Real_estate = st.sidebar.selectbox("Har du eiendom?", ["Ja", "Nei"])
    Ansatte = st.sidebar.slider("Antall ansatte", float(training_data.NoEmp.min()), float(training_data.NoEmp.max()),
                                float(training_data.NoEmp.mean()))
    Lanetermin = st.sidebar.slider("Lånetermin", float(training_data.Term.min()),
                                   float(training_data.Term.max()),
                                   float(training_data.Term.mean()))
    DisbursementGross = st.sidebar.slider("Disbursement", float(training_data.DisbursementGross.min()),
                                          float(training_data.DisbursementGross.max()),
                                          float(training_data.DisbursementGross.mean()))
    Lanemengde = st.sidebar.slider("Lånemengde", float(training_data.GrAppv.min()),
                                   float(training_data.GrAppv.max()), float(training_data.GrAppv.mean()))
    Sikkerhetsnett = st.sidebar.slider("Sikkerhetsnett", float(training_data.SBA_Appv.min()),
                                       float(training_data.SBA_Appv.max()),
                                       float(training_data.SBA_Appv.mean()))
    Ratio = Sikkerhetsnett / Lanemengde

    data = {
        "Eiendom": Real_estate,
        "Ansatte": Ansatte,
        "Lånetermin": Lanetermin,
        "DisbursementGross": DisbursementGross,
        "Lånemengde": Lanemengde,
        "Sikkerhetsnett": Sikkerhetsnett,
        "Ratio": Ratio
    }
    features = pd.DataFrame(data, index=[0])

    return features


# # In[25]:
#
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=21)
#
#
#
# # In[26]:
#
#
# # Scaling the data with StandardScaler()
#
sc = StandardScaler()
sc.fit(X_train)
#
#
# # Transform (standardise) both X_train and X_test with mean and STD from
# # training data
#
X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)
#
#
pred_user = brukerverdier()
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
# st.dataframe(data=pred_user, width=1200, height=768)
# st.dataframe(pred_user)
st.write(""" ## Se om de fortsatt får lån dersom man endrer noen parametere""")
st.write(""" ### Dine nye parametere""")
st.table(pred_user)
# st.write(pred_user)

pred_user["Eiendom"].replace({"Ja": 1, "Nei": 0}, inplace=True)
# If button is pressed
if st.button("Submit"):
    # Unpickle classifier
    ann = models.load_model("tin100-ann-model.h5")
    pred_streamlit = ann.predict(pred_user)
    if pred_streamlit == 0:
        st.write("Basert på dine oppgitte data er søknaden om lån godkjent ")
    else:
        st.write("Basert på dine oppgitte data er søknaden om lån avslått ")
# # In[27]:
#
# # In[28]:
#

