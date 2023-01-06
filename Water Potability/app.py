import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


with open('modelo.pickle','rb') as arquivo:
    modelo = pickle.load(arquivo)

#Title
st.title('Python Machine Learning APP')

st.subheader('Choose the options below to perform the prediction')

st.markdown('Please enter only numerical values')
ph = st.number_input('Value of PH')
Hardness = st.number_input('Value of Hardness')
Solids = st.number_input('Value of Solids')
Chloramines = st.number_input('Value of Chloramines')
Sulfate = st.number_input('Value of Sulfate')
Conductivity = st.number_input('Value of Conductivity')
Organic_carbon = st.number_input('Value of Organic_carbon')
Trihalomethanes = st.number_input('Value of Trihalomethanes')
Turbidity = st.number_input('Value of Turbidity')

Dados = {'ph':ph,'Hardness':Hardness,'Solids':Solids,'Chloramines':Chloramines,'Sulfate':Sulfate,
         'Conductivity':Conductivity,'Organic_carbon':Organic_carbon,'Trihalomethanes':Trihalomethanes,
         'Turbidity':Turbidity}


click = st.button('Fazer previsao')

if click:
    Dados_df = pd.DataFrame([Dados])
    Dados_df_sc = sc.fit_transform(Dados_df)
    y_pred = modelo.predict(Dados_df_sc)[0]
    y_pred = round(y_pred,2)
    st.write(y_pred)
