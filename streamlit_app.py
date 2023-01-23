#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import layers, models

import streamlit as st

#
# # In[5]:

st.set_page_config(layout="wide")
st.write(""" # Automatisering av låneprosessen for firmaer""")

st.sidebar.title("Vennligst fyll inn opplysningene dine her:")
#
#
#
# # In[6]:
#
#
st.sidebar.header("Inndata verdier")

st.write("---")
"""
Velkommen kjære bankkunde! \n
Hvis du er interessert i å se om et firma kan betjene et lån, kan du prøve dette programmet her.
Det eneste du må gjøre er å fylle inn opplysningene dine, så gjør programmet resten for deg.
Husk at dette bare er estimatorer basert på data hentet fra tidligere søkere. \n
Håper dette gjør prosessen lettere.
"""


def brukerverdier():
    Real_estate = st.sidebar.selectbox("Har du eiendom?", ["Ja", "Nei"])
    Ansatte = st.sidebar.slider("Antall ansatte", 0, 9999, 12)
    Lanetermin = st.sidebar.slider("Lånetermin", 0, 569, 111)
    Lanemengde = st.sidebar.slider("Lånemengde", 200, 5472000, 192687)
    Sikkerhetsnett = st.sidebar.slider("Sikkerhetsnett", 100, 5472000, 149489)
    Ratio = Sikkerhetsnett / Lanemengde

    data = {
        "Eiendom": Real_estate,
        "Ansatte": Ansatte,
        "Lånetermin": Lanetermin,
        "Lånemengde": Lanemengde,
        "Sikkerhetsnett": Sikkerhetsnett,
        "Ratio": Ratio,
    }
    features = pd.DataFrame(data, index=[0])

    return features


#
#
#
# # In[26]:
#
#
# # Scaling the data with StandardScaler()
#
#
#
# # Transform (standardise) both X_train and X_test with mean and STD from
# # training data
#
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
    if pred_streamlit == 1:
        st.write("Basert på dine oppgitte data er søknaden om lån avslått ")
    else:
        st.write("Basert på dine oppgitte data er søknaden om lån godkjent ")
# # In[27]:
#
# # In[28]:
