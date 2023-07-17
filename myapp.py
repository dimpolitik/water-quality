# -*- coding: utf-8 -*-

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
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import seaborn as sns

# run app

#streamlit run my_app_new.py
# https://dimpolitik-water-quality-myapp-yws3vl.streamlit.app/

st.set_page_config(layout="wide")

@st.cache_data
def preprocess_dataset(file, file_p, file_t, variables_c, response, case_study):
    df_van = pd.read_csv(file)
    
    df_van = df_van.rename(columns={'Date.x': 'Date'})
    
    vc0 = []
    vc0 = variables_c
    
    #msno.matrix(df_van)
    
    df_p = pd.read_csv(file_p)
    df_t = pd.read_csv(file_t)
    df_m = pd.merge(df_van, df_p[['Site','Date','Lat','Lon','Prec1','Prec7','Prec30','MaxP7','MaxP30']], on = ['Site','Date','Lat','Lon'])
    df_m = pd.merge(df_m, df_t[['Site','Date','Lat','Lon','Temp1','Temp7','Temp30','MaxT1','MaxT7','MaxT30','MinT1','MinT7','MinT30']], on = ['Site','Date','Lat','Lon'])
    vc = vc0 + [response]
    df = df_m[vc]   
     
    if case_study == 'classes_3':
        df[response] = df[response].replace(['BAD','POOR'],'POOR')
        df[response] = df[response].replace(['HIGH','GOOD'],'HIGH')
    else:
        df[response] = df[response].replace(['BAD','POOR', 'MODERATE'],'POOR')
        df[response] = df[response].replace(['HIGH','GOOD'],'HIGH')
    
    
    df['Slope (%)'] = df['Slope (%)'].fillna(df['Slope (%)'].median())
    df['Dist. To Source (m)'] = df['Dist. To Source (m)'].fillna(df['Dist. To Source (m)'].median())
    df['Salinity'] = df['Salinity'].fillna(df['Salinity'].median())
    
    # drop duplicated columns
    df = df.loc[:,~df.columns.duplicated()]
    
    # replace outliers
    vars_with_outliers = ['Conductivity',
                             'Nitrite', 
                             'Nitrate',
                             'Ammonium', 
                             'TP', 
                             'Salinity', 
                             'Elevation',]
    #for v in vars_with_outliers:
    #    high_vals = np.percentile(df[v], 95)
    #    median = float(df[v].median())
    #    df[v] = np.where(df[v] > high_vals, median, df[v])
    
    print('Initial shape', df.shape)
    df_ml = df.dropna() #subset=response)
    
    print('After dron na', df_ml.shape)
    #df_ml[logv] = (df_ml[logv]).apply(np.sqrt)  # (df_ml[logv]+1).apply(np.log10)       
    return df_ml, df_van
     
   
def split_dataset(df, response):
    predictors0 = df.drop(columns=response)   

    predictors = pd.get_dummies(predictors0, columns = ['Year'])
    
    labelencoder_season = LabelEncoder()
    predictors['Season'] = labelencoder_season.fit_transform(predictors['Season'])
    
    labelencoder_wd = LabelEncoder()
    predictors['WD'] = labelencoder_wd.fit_transform(predictors['WD'])

    predictors = predictors.astype({"Year_2018" : int, "Year_2019": int, "Year_2020": int, "Year_2021": int})
    
    #predictors['Year'] = predictors['Year'].astype(str)      
    #predictors = predictors.drop('Year', axis=1, inplace=True)
    
    X = predictors
    
    labelencoder = LabelEncoder()
    df[response] = labelencoder.fit_transform(df[response])
    y = df[response].values  
    indices = np.arange(predictors.shape[0])
    
    df_wiz = predictors.copy()
    df_wiz[response] = y
    
    # Split dataset into Train and Test
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, 
                                                                                     y,
                                                                                     indices,
                                                                                     test_size = 0.2, 
                                                                                     stratify=y, 
                                                                                     random_state = 7)
       
       
    return X_train, X_test, y_train, y_test, predictors, indices_train, indices_test, X, y, df_wiz

                    
def load_model(response):
    
    xgb_model = XGBClassifier()
    
    #models = [
    #     'xgb_MI_QUALITY_classes_2.json',
    #     'xgb_Diat_QUALITY_classes_2.json'
    #     'xgb_Fish_QUALITY_classes_2.json'
    #     'xgb_Total_QUALITY_classes_2.json']
    
    if response == 'MI_QUALITY':
        xgb_model.load_model('xgb_MI_QUALITY_classes_2.json')
    elif response == 'Diat_QUALITY':
        xgb_model.load_model( 'xgb_Diat_QUALITY_classes_2.json')
    elif response == 'FISH_QUALITY':
        xgb_model.load_model('xgb_Fish_QUALITY_classes_2.json')
    elif response == 'Total Quality':
        xgb_model.load_model('xgb_Total_QUALITY_classes_2.json')
    
    return xgb_model

responses = ['-', 'MI_QUALITY', 'Diat_QUALITY', 'FISH_QUALITY', 'Total Quality']

f = './dataset_clear.csv'
f_precip = './precipitation_era5-land.txt'
f_temp = './temperature_era5-land.txt'
     
cs = 'classes_2'

var = ['Season', 
            'Year',
            'WD',  # correlated
            'Conductivity',
           #'Turbidity',  # correalted
            #'BOD', # correlated
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
            #'Int. Type R-M'
            ] 

#meteo_vars = ['Prec1','Prec7','Prec30','MaxP7','MaxP30']
meteo_vars = ['Prec1']

#temp_vars = ['Temp1','Temp7','Temp30','MaxT1','MaxT7','MaxT30','MinT1','MinT7','MinT30']
temp_vars = ['Temp1']

variables_input = var + meteo_vars + temp_vars


######################### Webpage set up ############################
st.title('What-if-scenarios for predicting the ecological status of Greek rivers')
st.markdown("In this study, we developed a series of classification models based on XGBoost for predicting the ecological status of Greek rivers by using  machine learning. Three  indices were tested, namely the MI quality (MiQ), Diatoms quality (DiQ) and fish quality (FiQ). Physico-chemical parameters reflecting both water quality and hydromorphological status, collected from several river sites in Greece, were used as explanatory variables for the algorithms.")

st.markdown("The ecological status of the rivers was defined through three indices.")

#st.markdown("<h1 style='text-align: center; color: white;'>My Streamlit App</h1>", unsafe_allow_html=True)

expander = st.expander("See workflow")
expander.image('workflow.JPG', caption='Sunrise by the mountains')

st.markdown(" ")

st.write("You can use this tool to predict the ecological status and answer what-if questions such as")
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
        st.sidebar.subheader("Select index")
        response = st.sidebar.selectbox("Ecological index", responses)
        
        if response not in '-':  
            model = load_model(response)
            df_tq, df_vanilla = preprocess_dataset(f, f_precip, f_temp, variables_input, response, cs)
            
            
            X_train, X_test, y_train, y_test, predictors, indices_train, indices_test, X_all, y_all, df_wiz = split_dataset(df_tq, response)
            
            X_train_crop = X_train.drop(['Lon', 'Lat'], axis=1)
            X_test_crop = X_test.drop(['Lon', 'Lat'], axis=1)
            y_pred = model.predict(X_test_crop)
            X_test['Prediction'] = y_pred
            
            X_test['Prediction'] = X_test['Prediction'].replace([0], 'Pass')
            X_test['Prediction'] = X_test['Prediction'].replace([1], 'Fail')
                    
            X_test['Ground'] = y_test
            
            X_test['Ground'] = X_test['Ground'].replace([0], 'Pass')
            X_test['Ground'] = X_test['Ground'].replace([1], 'Fail')
            
            #st.sidebar.subheader("Pick a random data instance")
            # GAMATO: https://discuss.streamlit.io/t/if-syntax-error-while-attempting-to-incorporate-in-a-st-select-slider/34241/3
                
            st.subheader("Scenario (Input columns and % change)")
            st.markdown("Increase/Decrease (in %) to make a new prediction")
            
            # https://blog.streamlit.io/introducing-submit-button-and-forms/
            # https://discuss.streamlit.io/t/form-and-submit-button-in-sidebar/12436/
            # for i in range(3):
            #var_number = i
            #slider.append(st.slider(‘Change variable value:’, s_min, s_max, default_cal,
            #key=“sld_%d” % var_number ))         
                   
            
            #i=0; var0 = st.slider(var[i], -50, 50, 0, 1, key='my_slider0')           
            #i=1; var1 = st.slider(var[i], -50, 50, 0, 1, key='my_slider1')
            #i=2; var2 = st.slider(var[i], -50, 50, 0, 1, key='my_slider2')
            i=3; var3 = st.slider(var[i], -50, 50, 0, 1, key='my_slider3')
            i=4; var4 = st.slider(var[i], -50, 50, 0, 1, key='my_slider4')
            i=5; var5 = st.slider(var[i], -50, 50, 0, 1, key='my_slider5')
            i=6; var6 = st.slider(var[i], -50, 50, 0, 1, key='my_slider6')
            i=7; var7 = st.slider(var[i], -50, 50, 0, 1, key='my_slider7')
            #i=8; var8 = st.slider(var[i], -50, 50, 0, 1, key='my_slider8')
            i=9; var9 = st.slider(var[i], -50, 50, 0, 1, key='my_slider9')
            #i=10; var10 = st.slider(var[i], -50, 50, 0, 1, key='my_slider10')        
            i=11; var11 = st.slider(var[i], -50, 50, 0, 1, key='my_slider11')
            i=12; var12 = st.slider(var[i], -50, 50, 0, 1, key='my_slider12')
            i=13; var13 = st.slider(var[i], -50, 50, 0, 1, key='my_slider13')
           # i=14; var14 = st.slider(var[i], 0, 100, 0, 1, key='my_slider14')
            scenario = np.array([#st.session_state.my_slider0, 
                                 #st.session_state.my_slider1, 
                                 #st.session_state.my_slider2, 
                                 st.session_state.my_slider3, 
                                 st.session_state.my_slider4, 
                                 st.session_state.my_slider5, 
                                 st.session_state.my_slider6, 
                                 st.session_state.my_slider7, 
                                 #st.session_state.my_slider8, 
                                 st.session_state.my_slider9, 
                                 #st.session_state.my_slider10, 
                                 st.session_state.my_slider11, 
                                 st.session_state.my_slider12, 
                                 st.session_state.my_slider13]) 
                                 #st.session_state.my_slider14])
            
            
            w4 = st.form_submit_button('Predict')
       
if response not in '-': 
    st.subheader("Baseline and Scenario")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('Baseline-model')
        fig = plt.figure(figsize = (6,6))  
        datafile = 'greece.jpg'
        img = plt.imread(datafile)
        sns.scatterplot('Lon', 'Lat', data=X_test, hue='Prediction', zorder=1).set(title='Model')
        plt.imshow(img, zorder=0, extent=[19.25, 28.1, 34.5, 42])      

        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)
    with col2:   
        st.markdown('Baseline-Ground Truth')

        fig = plt.figure(figsize = (6,6))  
        datafile = 'greece.jpg'
        img = plt.imread(datafile)
        #plt.imshow(img, zorder=0, extent=[19.25, 28.1, 34.5, 42])
        #plt.scatter(X_test['Lon'], X_test['Lat'], s=25, zorder = 1) #, c=X_test_crop['predict'].map(colors))
        #plt.text(dfs['Lon']-0.2, dfs['Lat']+0.1, random_inst, fontsize = 12, color = "red")   
        
        sns.scatterplot('Lon', 'Lat', data=X_test, hue='Ground', zorder = 1).set(title='Truth')
        plt.imshow(img, zorder=0, extent=[19.25, 28.1, 34.5, 42])
        
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)
        
    if w4:
        with col3:
            st.markdown('Scenario-model')

            fig = plt.figure(figsize = (6,6))  
            #st.subheader("Scenario")
            scenario_exp = np.expand_dims(scenario, axis=0)
            
            X_test_vanilla = X_test_crop.copy() 
           
            X_test_crop.iloc[:,2] = X_test_crop.iloc[:,2]* (100+scenario[0])/100 # conductivity 
            X_test_crop.iloc[:,3] = X_test_crop.iloc[:,3]* (100+scenario[1])/100 # Nitrite
            X_test_crop.iloc[:,4] = X_test_crop.iloc[:,4]* (100+scenario[2])/100 # Nitrate
            X_test_crop.iloc[:,5] = X_test_crop.iloc[:,5]* (100+scenario[3])/100 # Ammonium
            X_test_crop.iloc[:,6] = X_test_crop.iloc[:,6]* (100+scenario[4])/100 # TP        
            X_test_crop.iloc[:,7] = X_test_crop.iloc[:,7]* (100+scenario[5])/100 # Salinity 
            X_test_crop.iloc[:,8] = X_test_crop.iloc[:,8]* (100+scenario[6])/100 # Elevation
            X_test_crop.iloc[:,9] = X_test_crop.iloc[:,9]* (100+scenario[7])/100 # Distance 
            X_test_crop.iloc[:,10] = X_test_crop.iloc[:,10]* (100+scenario[8])/100 # Slope
            
            X_test_new_prediction = X_test_crop.copy()
            
            pred_new = model.predict(X_test_crop)
            X_test['New prediction'] = pred_new
            
            X_test['New prediction'] = X_test['New prediction'].replace([0], 'Pass')
            X_test['New prediction'] = X_test['New prediction'].replace([1], 'Fail')
            
            sns.scatterplot('Lon', 'Lat', data=X_test, hue='New prediction', zorder = 1).set(title='New prediction')
            plt.imshow(img, zorder=0, extent=[19.25, 28.1, 34.5, 42])
            
            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.image(buf)


        st.markdown("Baseline values (4 first rows are shown)")
        st.write(X_test_vanilla.head(4))  

        st.markdown("Scenario values (4 first rows are shown)")
        st.write(X_test_new_prediction.head(4))          
