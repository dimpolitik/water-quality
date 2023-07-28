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
import gmaps
from ipywidgets import embed
import streamlit.components.v1 as components

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
        xgb_model.load_model('xgb_Fish_QUALITY_classes_2n.json')
    elif response == 'Total Quality':
        xgb_model.load_model('xgb_Total_QUALITY_classes_2n.json')
    
    return xgb_model 

def new_york():
    # https://discuss.streamlit.io/t/how-to-show-gmaps-object-from-google-maps-api-on-streamlit/24588/5
    gmaps.configure(api_key='AIzaSyChoXuIdG6ZoKUA1XEGr5pdclI1ToPHE1U')
    # Plot coordinates
    coordinates = (23.75, 38)
    _map = gmaps.figure(center=coordinates, zoom_level=12)

    # Render map in Streamlit
    snippet = embed.embed_snippet(views=_map)
    html = embed.html_template.format(title="", snippet=snippet)
    return components.html(html, height=500,width=500)

responses_back = ['-', 'MI_QUALITY', 'Diat_QUALITY', 'FISH_QUALITY', 'Total Quality']
responses = ['-', 'Macroinvertebrates', 'Diatoms', 'Fish', 'Physicochemical quality']


f = './dataset_clear.csv'
f_precip = './precipitation_era5-land.txt'
f_temp = './temperature_era5-land.txt'
     
cs = 'classes_2'

var0 = ['Season', 
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

show_vars = ['Conductivity',
            'Nitrite', 
            'Nitrate',
            'Ammonium', 
            'TP', 
            'Salinity', 
            'Prec1',
            'Temp1'
                    ] 

#meteo_vars = ['Prec1','Prec7','Prec30','MaxP7','MaxP30']
meteo_vars = ['Prec1']

#temp_vars = ['Temp1','Temp7','Temp30','MaxT1','MaxT7','MaxT30','MinT1','MinT7','MinT30']
temp_vars = ['Temp1']

var = var0 + meteo_vars + temp_vars

######################### Webpage set up ############################
st.title('What-if-scenarios for predicting the ecological status of Greek rivers')
st.markdown("In this study, we developed a series of classification models based on XGBoost for predicting the ecological status of Greek rivers by using  machine learning. Three  indices were tested, namely the MI quality (MiQ), Diatoms quality (DiQ) and fish quality (FiQ). Physico-chemical parameters reflecting both water quality and hydromorphological status, collected from several river sites in Greece, were used as explanatory variables for the algorithms.")

st.markdown("The ecological status of the rivers was defined through three indices.")

##st.markdown("<h1 style='text-align: center; color: white;'>My Streamlit App</h1>", unsafe_allow_html=True)

#expander = st.expander("See workflow")
#expander.image('workflow.JPG', caption='Sunrise by the mountains')

st.markdown(" ")

st.write("You can use this tool to predict the ecological status and answer what-if questions such as")
st.markdown("- What if we reduce pH?")
st.markdown("- Item 2")
st.markdown("- Item 3")

with st.form(key ='Form1'):
    with st.sidebar:
        st.sidebar.image('urk-fishing-trawlers.jpg')
        st.sidebar.subheader("Select index")
        response = st.sidebar.selectbox("Ecological index", responses)
        
        if response == 'Macroinvertebrates':
            response_back = responses_back[1]
        elif response == 'Diatoms':
            response_back = responses_back[2]
        elif response == 'Fish':
            response_back = responses_back[3]
        if response == 'Physicochemical quality':
            response_back = responses_back[4]
        
        if response not in '-':  
            model = load_model(response_back)
            df_tq, df_vanilla = preprocess_dataset(f, f_precip, f_temp, var, response_back, cs)
            
            
            X_train, X_test, y_train, y_test, predictors, indices_train, indices_test, X_all, y_all, df_wiz = split_dataset(df_tq, response_back)
            
            X_train_crop = X_train.drop(['Lon', 'Lat'], axis=1)
            X_test_crop = X_test.drop(['Lon', 'Lat'], axis=1)
            y_pred = model.predict(X_test_crop)
            X_test['Prediction'] = y_pred
            X_test['Before'] = y_pred
            X_test['Prediction'] = X_test['Prediction'].replace([0], 'Pass')
            X_test['Prediction'] = X_test['Prediction'].replace([1], 'Fail')
                    
            #X_test['Ground'] = y_test
            #X_test['Ground'] = X_test['Ground'].replace([0], 'Pass')
            #X_test['Ground'] = X_test['Ground'].replace([1], 'Fail')
            
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
                         
            i=3; var3 = st.slider(var[i], -100, 100, 0, 1, key='my_slider3')
            i=4; var4 = st.slider(var[i], -100, 100, 0, 1, key='my_slider4')
            i=5; var5 = st.slider(var[i], -100, 100, 0, 1, key='my_slider5')
            i=6; var6 = st.slider(var[i], -100, 100, 0, 1, key='my_slider6')
            i=7; var7 = st.slider(var[i], -100, 100, 0, 1, key='my_slider7')
          
            i=9; var9 = st.slider(var[i], -100, 100, 0, 1, key='my_slider9')
            
            i=14; var14 = st.slider(var[i], -100, 100, 0, 1, key='my_slider14')
            i=15; var14 = st.slider(var[i], -100, 100, 0, 1, key='my_slider15')

            scenario = np.array([
                                 st.session_state.my_slider3, 
                                 st.session_state.my_slider4, 
                                 st.session_state.my_slider5, 
                                 st.session_state.my_slider6, 
                                 st.session_state.my_slider7, 
                                 st.session_state.my_slider9, 
                                 st.session_state.my_slider14, 
                                 st.session_state.my_slider15 
                                                            ]) 
                                 #st.session_state.my_slider14])
            
            
            w4 = st.form_submit_button('Predict')
       
if response not in '-': 
    st.subheader("Baseline and Scenario")
    
    if response == responses[1]:
        st.write('<p style="font-size:18px; color:red;">Most influential factors for Macroinvertebrates are ???</p>',
                                                                 unsafe_allow_html=True)
    elif response == responses[2]:
        st.write('<p style="font-size:18px; color:red;">Most influential factors for Diatoms are ???</p>',
                                                                 unsafe_allow_html=True)
    elif response == responses[3]:
       st.write('<p style="font-size:18px; color:red;">Most influential factors for Fish are ???</p>',
                                                                unsafe_allow_html=True)
    elif response == responses[4]:
        st.write('<p style="font-size:18px; color:red;">Most influential factors for Physicochemical quality are ???</p>',
                                                                 unsafe_allow_html=True)
        
    col1, col2 = st.columns(2)
    with col1:
        #st.markdown('Baseline-model')
        fig = plt.figure(figsize = (6,6))  
        datafile = 'greece.jpg'
        img = plt.imread(datafile)
        
        color_dict1 = dict({'Pass': '#1f77b4',
                           'Fail': 'orange', 
                                       })
        
        sns.scatterplot(data=X_test, x='Lon', y='Lat', hue='Prediction', palette= color_dict1).set(title='Baseline')
        plt.imshow(img, zorder=0, extent=[19.25, 28.1, 34.5, 42])      

        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)
    #with col2:   
        
        #st.markdown('Baseline-Ground Truth')

        #fig = plt.figure(figsize = (6,6))  
        #datafile = 'greece.jpg'
        #img = plt.imread(datafile)
       # 
        #sns.scatterplot('Lon', 'Lat', data=X_test, hue='Ground').set(title='Truth')
        #plt.imshow(img, zorder=0, extent=[19.25, 28.1, 34.5, 42])
        
        #buf = BytesIO()
        #fig.savefig(buf, format="png")
        #st.image(buf)
        
    if w4:
        with col2:
            #st.markdown('Scenario-model')

            fig = plt.figure(figsize = (6,6))  
            #st.subheader("Scenario")
            scenario_exp = np.expand_dims(scenario, axis=0)
            
            X_test_vanilla = X_test_crop.copy() 
            #st.write(X_test_crop)
            X_test_crop.iloc[:,2] = X_test_crop.iloc[:,2]* (100+scenario[0])/100 # conductivity 
            X_test_crop.iloc[:,3] = X_test_crop.iloc[:,3]* (100+scenario[1])/100 # Nitrite
            X_test_crop.iloc[:,4] = X_test_crop.iloc[:,4]* (100+scenario[2])/100 # Nitrate
            X_test_crop.iloc[:,5] = X_test_crop.iloc[:,5]* (100+scenario[3])/100 # Ammonium
            X_test_crop.iloc[:,6] = X_test_crop.iloc[:,6]* (100+scenario[4])/100 # TP        
            X_test_crop.iloc[:,7] = X_test_crop.iloc[:,7]* (100+scenario[5])/100 # Salinity 
            X_test_crop.iloc[:,11] = X_test_crop.iloc[:,11]* (100+scenario[6])/100 # Prec
            X_test_crop.iloc[:,12] = X_test_crop.iloc[:,12]* (100+scenario[7])/100 # Temp 
           
            
            X_test_new_prediction = X_test_crop.copy()
            
            pred_new = model.predict(X_test_crop)
            X_test['New prediction'] = pred_new
            X_test['After'] = pred_new
            
            idx = X_test.index[X_test['Before'] != X_test['After']]
            
            color_dict2 = dict({'Pass': '#1f77b4',
                               'Fail': 'orange', 
                               'Fail-> Pass': 'green',
                               'Pass-> Fail': 'red'})
            
            X_test['New prediction'][idx] = X_test['Before'][idx] - 2* X_test['After'][idx]
            
            X_test['New prediction'] = X_test['New prediction'].replace([0], 'Pass')
            X_test['New prediction'] = X_test['New prediction'].replace([1], 'Fail')
            X_test['New prediction'] = X_test['New prediction'].replace([2], 'Fail')
            X_test['New prediction'] = X_test['New prediction'].replace([-2], 'Pass-> Fail')
            X_test['New prediction'] = X_test['New prediction'].replace([2], 'Fail-> Pass')
             
            sns.scatterplot(data=X_test, x='Lon', y='Lat', hue='New prediction', palette= color_dict2).set(title='Scenario')
            plt.imshow(img, zorder=0, extent=[19.25, 28.1, 34.5, 42])
            
            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.image(buf)

        

        col1, col2 = st.columns(2)
        with col1:           
            st.markdown("Baseline values (5 first rows are shown)")
            d1 = X_test_vanilla[show_vars] 
            #d1.round(4)
            #st.write(d1)
            st.dataframe(d1.head(5).set_index(d1.columns[0]))
            #st.markdown(d1.head(5).style.hide(axis="index").to_html(), unsafe_allow_html=True)
        with col2:
            #st.write(X_test_vanilla[show_vars].head(5))  
            st.markdown("Scenario values (5 first rows are shown)")
            d2 = X_test_new_prediction[show_vars]
            
            st.dataframe(d2.head(5).set_index(d2.columns[0]))
