# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 11:20:59 2021
@author: ΔΗΜΗΤΡΗΣ
"""
# https://towardsdatascience.com/quickly-build-and-deploy-an-application-with-streamlit-988ca08c7e83
# https://www.analyticsvidhya.com/blog/2021/06/build-web-app-instantly-for-machine-learning-using-streamlit/
# https://docs.streamlit.io/library/api-reference
# https://towardsdatascience.com/data-visualization-using-streamlit-151f4c85c79a
# https://carpentries-incubator.github.io/python-interactive-data-visualizations/07-add-widgets/index.html
# https://docs.streamlit.io/library/api-reference/layout/st.expander
# https://towardsdatascience.com/deploying-a-web-app-with-streamlit-sharing-c320c79ae350
# https://share.streamlit.io/dimpolitik/trawlers/main/myapp.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import shapefile as shp
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch
import math
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
import joblib

# run app

#streamlit run myapp.py

st.set_page_config(layout="wide")


@st.cache
def load_data():
    df = pd.read_excel("2streamlit.xlsx")
    labelencoder_season = LabelEncoder()
    df['Season'] = labelencoder_season.fit_transform(df['Season'])
    return df 

def load_variables(df, CLASS, response_var):
    # GAMATO: https://discuss.streamlit.io/t/if-syntax-error-while-attempting-to-incorporate-in-a-st-select-slider/34241/3
    if CLASS == 'High/Good':
        dfv = df.loc[ (df[response_var] == 'HIGH') | ( df[response_var] == 'GOOD')]    
        dfv = dfv.sample()   
        instance1 = dfv.copy()
        return dfv, instance1
    elif CLASS == 'Moderate':
        dfv = df.loc[(df[response_var] == 'MODERATE')]
        dfv = dfv.sample()
        instance1 = dfv.copy()
        return dfv, instance1
    elif CLASS == 'Poor/Bad':
        dfv = df.loc[ (df[response_var] == 'POOR') | (df[response_var] == 'BAD')] 
        dfv = dfv.sample()
        instance1 = dfv.copy()
        return dfv, instance1  
    else:
        dfv = 0*df.sample()
        return dfv, 1
                    
def load_model():
    m  = joblib.load('xgb_smote.pkl')
    return m

df = load_data()
model = load_model()

responses = ['-', 'MI_QUALITY', 'Diat_QUALITY', 'FISH_QUALITY']
classes1 = ['-', 'High/Good', 'Moderate', 'Poor/Bad']
classes = ['High/Good', 'Moderate', 'Poor/Bad']

var = ['Season', 
       'Year',
       'Conductivity',
       'Nitrite', 
       'Nitrate',
       'Ammonium', 
       'TP', 
       'Lat', 
       'Salinity', 
       'Lon', 
       'Elevation',
        'Dist. To Source (m)', 
        'Slope (%)', 
        'Prec7',
        'Temp7',
        'MI_QUALITY'
            ] 

######################### Webpage set up ############################
st.title('What-if-scenarios for predicting the ecological status of Greek rivers')
st.markdown("In this study, we developed a series of classification model based on XGBoost for predicting the ecological status of Greek rivers by using  machine learning. Three  indices were tested, namely the MI quality (MiQ), Diatoms quality (DiQ) and fish quality (FiQ). Physico-chemical parameters reflecting both water quality and hydromorphological status, collected from several river sites in Greece, were used as explanatory variables for the algorithms.")

st.markdown("The ecological status of the rivers was defined through three indices, namely M??? quality (MiQ), Diatoms quality (DiQ), Fish quality (FiQ) and Total quality (TotQ). Each index was categorized into five classes, namely High, Good, Moderate, Poor and Bad. These classes were defined based on ???? and represent different point of views in river quality. A short description for each index can be found in Table XXX.")

#st.markdown("<h1 style='text-align: center; color: white;'>My Streamlit App</h1>", unsafe_allow_html=True)

expander = st.expander("See workflow")
expander.image('workflow.JPG', caption='Sunrise by the mountains')

st.markdown(" ")

st.write("You can use this tool to predict the cological status and answer what-if questions such as")
st.markdown("- What if we reduce pH?")
st.markdown("- Item 2")
st.markdown("- Item 3")

st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    padding-left:40px;
}
</style>
''', unsafe_allow_html=True)

with st.form(key ='Form1'):
    with st.sidebar:
        st.sidebar.image('urk-fishing-trawlers.jpg')
        st.sidebar.subheader("Selections")
        response = st.sidebar.selectbox("Ecological index", responses)
        random_inst = st.sidebar.selectbox("Random data instance", classes1)
        dfs, instance = load_variables(df, random_inst, response)
        
        #st.sidebar.subheader("Pick a random data instance")
        # GAMATO: https://discuss.streamlit.io/t/if-syntax-error-while-attempting-to-incorporate-in-a-st-select-slider/34241/3
            
        st.subheader("Scenario (Input columns and change)")
        st.markdown("You can change the values to make a new prediction")
        
        # https://blog.streamlit.io/introducing-submit-button-and-forms/
        # https://discuss.streamlit.io/t/form-and-submit-button-in-sidebar/12436/
        # for i in range(3):
        #var_number = i
        #slider.append(st.slider(‘Change variable value:’, s_min, s_max, default_cal,
        #key=“sld_%d” % var_number ))         
               
        i=0; var0 = st.slider(var[i], int(np.min(df[var[i]])), int(np.max(df[var[i]])), int(dfs[var[i]]), 1, key='my_slider0')           
        i=1; var1 = st.slider(var[i], int(np.min(df[var[i]])), int(np.max(df[var[i]])), int(dfs[var[i]]), 1, key='my_slider1')
        i=2; var2 = st.slider(var[i], float(np.min(df[var[i]])), float(np.max(df[var[i]])), float(dfs[var[i]]), 0.1, key='my_slider2')
        i=3; var3 = st.slider(var[i], float(np.min(df[var[i]])), float(np.max(df[var[i]])), float(dfs[var[i]]), 0.1, key='my_slider3')
        i=4; var4 = st.slider(var[i], float(np.min(df[var[i]])), float(np.max(df[var[i]])), float(dfs[var[i]]), 0.1, key='my_slider4')
        i=5; var5 = st.slider(var[i], float(np.min(df[var[i]])), float(np.max(df[var[i]])), float(dfs[var[i]]), 0.1, key='my_slider5')
        i=6; var6 = st.slider(var[i], float(np.min(df[var[i]])), float(np.max(df[var[i]])), float(dfs[var[i]]), 0.1, key='my_slider6')
        i=7; var7 = st.slider(var[i], float(np.min(df[var[i]])), float(np.max(df[var[i]])), float(dfs[var[i]]), 0.1, key='my_slider7')
        i=8; var8 = st.slider(var[i], float(np.min(df[var[i]])), float(np.max(df[var[i]])), float(dfs[var[i]]), 0.1, key='my_slider8')
        i=9; var9 = st.slider(var[i], float(np.min(df[var[i]])), float(np.max(df[var[i]])), float(dfs[var[i]]), 0.1, key='my_slider9')
        i=10; var10 = st.slider(var[i], float(np.min(df[var[i]])), float(np.max(df[var[i]])), float(dfs[var[i]]), 0.1, key='my_slider10')        
        i=11; var11 = st.slider(var[i], float(np.min(df[var[i]])), float(np.max(df[var[i]])), float(dfs[var[i]]), 0.1, key='my_slider11')
        i=12; var12 = st.slider(var[i], float(np.min(df[var[i]])), float(np.max(df[var[i]])), float(dfs[var[i]]), 0.1, key='my_slider12')
        i=13; var13 = st.slider(var[i], float(np.min(df[var[i]])), float(np.max(df[var[i]])), float(dfs[var[i]]), 0.1, key='my_slider13')
        i=14; var14 = st.slider(var[i], float(np.min(df[var[i]])), float(np.max(df[var[i]])), float(dfs[var[i]]), 0.1, key='my_slider14')
        scenario = np.array([st.session_state.my_slider0, 
                             st.session_state.my_slider1, 
                             st.session_state.my_slider2, 
                             st.session_state.my_slider3, 
                             st.session_state.my_slider4, 
                             st.session_state.my_slider5, 
                             st.session_state.my_slider6, 
                             st.session_state.my_slider7, 
                             st.session_state.my_slider8, 
                             st.session_state.my_slider9, 
                             st.session_state.my_slider10, 
                             st.session_state.my_slider11, 
                             st.session_state.my_slider12, 
                             st.session_state.my_slider13, 
                             st.session_state.my_slider14])
        #st.write(scenario)
        #w4 = st.sidebar.button("Predict")  
        w4 = st.form_submit_button('Predict')
       
if response not in '-' and random_inst not in '-':   
    v = instance.iloc[:,:-1].values
    v  = np.array(v.flatten())
    #v = [int(x) if type(x) == int else np.round(x,3) for x in v]
    default_vars = pd.DataFrame(v, index = var[:-1], columns = ['Default values'])
    col1, col2 = st.columns(2)
    with col1:
        fig = plt.figure(figsize = (5,8))  
        datafile = 'greece.jpg'
        img = plt.imread(datafile)
        plt.imshow(img, zorder=0, extent=[19.25, 28.6, 34.5, 42])
        plt.scatter(dfs['Lon'], dfs['Lat'], s=25, zorder = 1)
        plt.text(dfs['Lon']-0.2, dfs['Lat']+0.1, random_inst, fontsize = 12, color = "red")
        
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)
    with col2:        
        st.table(default_vars)
        
    if w4:
        scenario_exp = np.expand_dims(scenario, axis=0)
        pred = model.predict(scenario_exp)
                        
        if (pred == 0): pred = 'High/Good'
        if (pred == 1): pred = 'Moderate'  
        if (pred == 2): pred = 'Poor/Bad'
        
        st.write('New prediction is: ', pred)
        new_vars = pd.DataFrame(scenario, index = var[:-1], columns = ['New values'])
        st.table(new_vars)
               
         
