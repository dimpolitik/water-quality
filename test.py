import streamlit as st
import numpy as np
import matplotlib.pylab as plt
import time

def form_callback():
    st.write(st.session_state.my_slider)
    st.write(st.session_state.my_slider1)
    st.write(st.session_state.my_checkbox)

with st.form(key='my_form'):
    slider_input = st.slider('My slider', 0, 10, 5, key='my_slider')
    slider_input1 = st.slider('My slider', 1, 100, 20, key='my_slider1')
    checkbox_input = st.checkbox('Yes or No', key='my_checkbox')
    submit_button = st.form_submit_button(label='Submit', on_click=form_callback)