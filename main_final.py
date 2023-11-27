# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 19:07:35 2022

@author: Hyeonji Oh
"""

# import libraries needed
import streamlit as st
import streamlit_nested_layout

import pandas as pd
import numpy as np

from datetime import date, timedelta
from datetime import datetime
from dateutil.relativedelta import relativedelta

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import joblib
import seaborn as sns
import altair as alt 

import lifelines
import sksurv.ensemble

import pickle
import cloudpickle as cp
from urllib.request import Request, urlopen
import zipfile
import os


pd.set_option('display.width', 10) 
st.set_page_config(layout='wide')

# when using local files...
proact_train_set = pd.read_csv('data/X_and_Y_with_NaNs.csv')

aft = pd.read_pickle('data/aft1122.pkl')
cph = pd.read_pickle('data/cph1122.pkl')
    
train_aft = pd.read_csv('data/train_aft_1122.csv')
train_cph = pd.read_csv('data/train_cph_1122.csv')
train_rsf = pd.read_csv('data/train_rsf_1122.csv')
train_median = pd.read_csv('data/train_median_1122.csv')
 


#Feature selection results
aft_scale_list = ['Age', 'mean_R3_Respiratory_Insufficiency', 'mean_weight', 'diag_delta', 'mean_Q5_Cutting', 'mean_creatinine', 'mean_Q2_Salivation', 'mean_fvc', 'mean_Q3_Swallowing', 'mean_Q7_Turning_in_Bed', 'onset_delta', 'mean_bulbar']

cph_scale_list = ['mean_bulbar', 'onset_delta', 'mean_Q3_Swallowing', 'mean_fvc', 'mean_Q7_Turning_in_Bed', 'mean_Q2_Salivation', 'mean_creatinine', 'mean_weight', 'mean_ALSFRS_R_Total', 'diag_delta', 'mean_Q5_Cutting', 'mean_R3_Respiratory_Insufficiency', 'Age']


# Setting the Robust scaler
proact_train_set.drop(columns=['Unnamed: 0'], inplace = True)
X_y_NaN = proact_train_set.sort_values(by='SubjectID').copy()

aft_columns = X_y_NaN[aft_scale_list].columns
cph_columns = X_y_NaN[cph_scale_list].columns

aft_scaler = RobustScaler()
aft_scaler.fit(X_y_NaN[aft_scale_list])

cph_scaler = RobustScaler()
cph_scaler.fit(X_y_NaN[cph_scale_list])





   
def extract_and_load_pkl(zip_path, pkl_filename, extract_to='.'):
    # Extract the pkl file from the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract(pkl_filename, extract_to)

    pkl_path = os.path.join(extract_to, pkl_filename)

    # Load the extracted pkl file
    with open(pkl_path, 'rb') as file:
        data = pd.read_pickle(file)

    # Optionally, remove the extracted file if you don't need it
    os.remove(pkl_path)

    return data

rsf = extract_and_load_pkl('data/rsf1122.zip', 'rsf1122.pkl')

# Define a function which is needed later
def predict_rsf_percentile(data, percentile):
    result_per = rsf.predict_survival_function(data.to_numpy().reshape(1, -1), return_array = True)
    result_per = np.squeeze(result_per)
    time_result = pd.DataFrame({'time' : rsf.event_times_, 'p' : result_per })
    if time_result[time_result['p'] <= percentile].count()['time'] == 0:
      per = np.inf
    else:
      per = time_result[time_result['p'] <= percentile].iloc[0,0]
    
    return per

def find_percentile(value, array):
   # Calculate the percentile rank of the input value within the array
   percentile = np.sum(array < value) / len(array) * 100
   return percentile

def ordinal(n):
    return "%d%s" % (n, "th" if 4 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th"))


# Set title
ti_col1, ti_col2 = st.columns([3.15,0.76])
with ti_col1:
    st.title('Predicting Loss of Autonomy in Swallowing Function in ALS Patients')
    st.subheader('Web application ver. (※Not For Clinical Use)')
    st.subheader('')
with ti_col2:
    st.image('https://github.com/PROACT-team/2022-Final-scripts/assets/78291206/3c6b0278-157a-4185-b4dd-9fa90690fe58')
    tti_col1, tti_col2 = st.columns([1.3,1])
       
# Create Tabs (page shifting)
tab1, tab2 = st.tabs(['Run Prediction', 'About'])

# Tab style
st.markdown("""
<style>

	.stTabs [data-baseweb="tab-list"] {
		gap: 5px;
        font-size: 100px;
    }

	.stTabs [data-baseweb="tab"] {
		height: 40px;
        white-space: pre-wrap;
		background-color: #F9F9F9;
        font-size: 100px;
        margin:100;
        width: 200%;
		border-radius: 2px 2px 0px 0px;
		gap: 7px;
		padding-top: 0px;
		padding-bottom: 0px;
    }

	.stTabs [aria-selected="true"] {
  		background-color: None;
	}

</style>       
""", unsafe_allow_html=True)

css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1.6rem;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)


min_date = pd.to_datetime(date.today() - relativedelta(months=3))
default_alsfrs_raw  = pd.DataFrame(    [
       #{ "Date": min_date, "Q1": 4, "Q2": 3, "Q3": 3, 
        #"Q4": 2, "Q5": 3, "Q6": 3, "Q7": 3, 
        #"Q8": 3, "Q9": 3, "R1": 3, "R2": 3, "R3": 3},
       { "Date": pd.to_datetime(np.nan), "Q1":  np.nan, "Q2": np.nan , "Q3":  np.nan, 
        "Q4": np.nan , "Q5":  np.nan, "Q6": np.nan , "Q7":  np.nan, 
        "Q8": np.nan , "Q9": np.nan , "R1": np.nan , "R2": np.nan , "R3": np.nan },
       {"Date": np.nan, "Q1":  np.nan, "Q2": np.nan , "Q3":  np.nan, 
        "Q4": np.nan , "Q5":  np.nan, "Q6": np.nan , "Q7":  np.nan, 
        "Q8": np.nan , "Q9": np.nan , "R1": np.nan , "R2": np.nan , "R3": np.nan }
       ])

if 'reset_clicked' not in st.session_state:
    st.session_state.reset_clicked = False
if 'reset_counter' not in st.session_state:
    st.session_state['reset_counter'] = 0    
if 'edited_alsfrs_raw' not in st.session_state:
    st.session_state['edited_alsfrs_raw'] = default_alsfrs_raw.copy()
    
# Your original code
timestamp = pd.Timestamp(date.today())
date_string = timestamp.strftime("%Y-%m-%d")

with tab1:
 # Run prediction을 누르면 표시될 내용 
 st.title('')
 
 h_col1, h_col2= st.columns([4.1,6])
 with h_col1:
  st.subheader('1. Patient Info') 
  st.write('Please fill in the input values. Use default values for unknown data') 
  with st.expander("Patient Info", expanded=True):   
   data_editor_key = f"data_editor_{st.session_state['reset_counter']}"
   
  # Patient feature input box
   with st.form(key='input_box'):
      # Static features
      st.write('Use default values for unknown data')
      st.subheader('Demographics & ALS history')     
      col1, col2 = st.columns([1,1.4])
      with col1:
          help_onset = (f'Enter the value as of today ({date_string}).\n'
                        'This will be automatically aligned with your latest ALSFRS record date.')
          Age = st.text_input(label='Age (years)', value=56, help = help_onset)
          onset_site = st.selectbox('Onset region', ['Non-Bulbar', 'Bulbar'])
      with col2:
          
          onset_delta = st.text_input(label='Time from symptom onset (months)', value=19, help=help_onset)
          diag_delta = st.text_input(label='Time from Diagnosis (months)', value=8, help=help_onset)
          
       
      fcol1, fcol2, fcol3 = st.columns(3)
      with fcol1:
          st.subheader('FVC')  
          mean_fvc = st.text_input(label='Forced Vital Capacity (%)', value=84)
      with fcol2:
          st.subheader('Creatinine')  
          mean_creatinine = st.text_input(label='Serum creatinine (umol/L)', value=67)
      with fcol3:
          st.subheader('Weight')
          help_weight = '''Enter the average value during the follow-up'''
          mean_weight = st.text_input(label='Weight (kg)', value=79, help=help_weight)
          
      if onset_site == 'Bulbar':
          onset_site_calculated = 1
      else :
          onset_site_calculated = 0
      Age_current = (float(Age)-float(15))//5
      onset_delta_current = (float(onset_delta)-3.0)  
      diag_delta_current = (float(diag_delta)-3.0)  
      
      # Time-resolved features
      st.subheader('ALSFRS-R over the past 3 months')
      
      st.write("""
**ALSFRS-R items list**
""")
      fre_col1, fre_col2, fre_col3 = st.columns([1,1.25,1.3])
      with fre_col1:
          st.write("""
                  **Q1**: Speech
                  <br>
                  **Q2**: Salivation
                  <br>
                  **Q3**: Swallowing
                  <br>
                  **Q4**: Handwriting
                  """, unsafe_allow_html=True)
      with fre_col2:
          st.write("""
                  **Q5**: Cutting
                  <br>
                  **Q6**: Dressing and Hygiene
                  <br>
                  **Q7**: Turning in Bed
                  <br>
                  **Q8**: Walking
                   """, unsafe_allow_html=True)
      with fre_col3:
          st.write("""
                  **Q9**: Climbing Stairs
                  <br>
                  **R1**: Dyspnea
                  <br>
                  **R2**: Orthopnea
                  <br>
                  **R3**: Respiratory Insufficiency
                  """, unsafe_allow_html=True)
          

      
      
            
      container1 = st.container()
      container2 = st.container()
      
      with container2:
       svcol1, svcol2 = st.columns([1.3,4])
       with svcol1:
        save_button = st.form_submit_button('Click to Save')
       with svcol2:
        reset_button = st.form_submit_button('Reset')
       if reset_button:
          # Increment the reset counter
          st.session_state.reset_clicked = True
          st.session_state['reset_counter'] += 1
          # Reset the dataframe
          st.session_state['edited_alsfrs_raw'] = default_alsfrs_raw.copy()
          with h_col1:
              st.experimental_rerun()
          
          
      with container1:
       
       help_input = '''**※ Data sufficiency**: At least 2 records are needed (Add more records if wanted)\n
**※ Date of visit**: Type in the form of YYYY-MM-DD or Click the calendar icon\n
**※ Follow-up duration**: should be in range of 1.5 ~ 3.0 months\n
**※ Error messages**: Try again with "Click to Save" if an error message appears\n'''
       st.markdown('**Please read the instructions first →**', help=help_input)
       
       edited_alsfrs_raw = st.data_editor(st.session_state['edited_alsfrs_raw'], 
                                          key=data_editor_key, num_rows="dynamic", 
                                          use_container_width=True, 
                                          column_config={
                                              "Date": st.column_config.DateColumn(
                                                  "Date",
                                                  min_value=date(1900, 1, 1),
                                                  max_value=date.today(),
                                                  format="YYYY-MM-DD",
                                                  step=1,
                                                  )}).query('index != "Example:"')
       

      edited_alsfrs_raw['ALSFRS_R_Total'] = edited_alsfrs_raw[["Q1", "Q2", "Q3", "Q4",
                                                              "Q5", "Q6", "Q7",
                                                              "Q8", "Q9", "R1", 
                                                              "R2", "R3"]].sum(axis=1)
      
      edited_alsfrs_raw['bulbar'] = edited_alsfrs_raw[['Q1', 'Q3']].sum(axis=1)
      
      # Create initial value
      mean_ALSFRS_R_Total = 0
      
      mean_bulbar = 0
      
      mean_Q1_Speech = 0
      
      mean_Q2_Salivation = 0
      
      mean_Q3_Swallowing = 0
      
      mean_Q5_Cutting = 0
      
      mean_Q7_Turning_in_Bed = 0
      
      mean_R3_Respiratory_Insufficiency = 0
      
      
      
      
        
      # Create error messages
      
      current_date = pd.Timestamp((date.today() - relativedelta(months=3)).strftime("%Y-%m-%d"))
      edited_alsfrs_raw['visit_date'] = pd.to_datetime(edited_alsfrs_raw['Date'])
      edited_alsfrs_raw['delta'] = (edited_alsfrs_raw['visit_date']- pd.to_datetime(date.today())) / np.timedelta64(1,'D') * 12/365
      edited_alsfrs_raw['feature_delta'] = edited_alsfrs_raw['delta']-(edited_alsfrs_raw['delta'].min())
      
      # Apply the CSS hack to the dataframe input
      st.markdown("""
                    <style>
                    /* metric styling */
                    div.css-3mmywe.e15ugz7a0{
			font-size: 3px;
            }
            </style>
                    """, unsafe_allow_html=True) 
      
      if not edited_alsfrs_raw.isna().any().any():
        if (edited_alsfrs_raw['visit_date'].max() - edited_alsfrs_raw['visit_date'].min()).days >= 45:
          model1 = LinearRegression().fit(edited_alsfrs_raw[['feature_delta']].values, edited_alsfrs_raw['ALSFRS_R_Total'].values)
          model2 = LinearRegression().fit(edited_alsfrs_raw[['feature_delta']].values, edited_alsfrs_raw['Q1'].values)
          model3 = LinearRegression().fit(edited_alsfrs_raw[['feature_delta']].values, edited_alsfrs_raw['Q2'].values)
          model4 = LinearRegression().fit(edited_alsfrs_raw[['feature_delta']].values, edited_alsfrs_raw['Q6'].values)
          model5 = LinearRegression().fit(edited_alsfrs_raw[['feature_delta']].values, edited_alsfrs_raw['Q8'].values)
          model6 = LinearRegression().fit(edited_alsfrs_raw[['feature_delta']].values, edited_alsfrs_raw['R3'].values)

          mean_ALSFRS_R_Total = round(edited_alsfrs_raw['ALSFRS_R_Total'].mean(),4)
          slope_ALSFRS_R_Total = round(model1.coef_[0],4)
              
          mean_bulbar = round(edited_alsfrs_raw['bulbar'].mean(),4)
              
          mean_Q1_Speech = round(edited_alsfrs_raw['Q1'].mean(),4)
          slope_Q1_Speech = round(model2.coef_[0],4)
              
          mean_Q2_Salivation = round(edited_alsfrs_raw['Q2'].mean(),4)
          slope_Q2_Salivation = round(model3.coef_[0],4)
              
          mean_Q3_Swallowing = round(edited_alsfrs_raw['Q3'].mean(),4)
          
          mean_Q5_Cutting = round(edited_alsfrs_raw['Q5'].mean(),4)
          
          slope_Q6_Dressing_and_Hygiene = round(model4.coef_[0],4)
              
          mean_Q7_Turning_in_Bed = round(edited_alsfrs_raw['Q7'].mean(),4)
              
          slope_Q8_Walking = round(model5.coef_[0],4)
              
          mean_R3_Respiratory_Insufficiency  = round(edited_alsfrs_raw['R3'].mean(),4)
          slope_R3_Respiratory_Insufficiency = round(model6.coef_[0],4)
          
      
      if mean_ALSFRS_R_Total == 0:
          onset_delta_calculated = onset_delta_current
          diag_delta_calculated = diag_delta_current
          Age_calculated = Age_current
          
      else:
          max_record = pd.Timestamp((edited_alsfrs_raw['visit_date'].max())).strftime("%Y-%m-%d")
          subtract = (pd.to_datetime((date.today())) - (pd.to_datetime(edited_alsfrs_raw['Date'].max())))/ np.timedelta64(1,'D') * 12/365
          onset_delta_calculated = round(onset_delta_current - subtract,1)
          diag_delta_calculated = round(diag_delta_current - subtract,1)
          
          
          max_record_date = pd.to_datetime(edited_alsfrs_raw['visit_date'].max())
          Age_calculated = Age_current - (datetime.now().year - max_record_date.year)
          
      if save_button:
          if edited_alsfrs_raw.isna().any().any():
              st.error("Error : Re-check if all values are filled.")
          # 2. Check for Follow-Up Duration
          elif (edited_alsfrs_raw['visit_date'].max() - edited_alsfrs_raw['visit_date'].min()).days < 45:
              st.error("Error : Follow-up duration is too short.")

          # 3. Check for Out-of-Date Data
          elif (edited_alsfrs_raw['visit_date'].max() - edited_alsfrs_raw['visit_date'].min()).days >92:
              st.error("Error : Follow-up duration is too long. Try removing any out-of-date data.")
          else:
              st.success("Your input has been successfully saved.")
              
      
      
 # Apply the CSS hack to the left expander
  st.markdown("""
                    <style>
                    /* metric styling */
                    div.css-ixecyn.en8akda1{
			background-color: #white;
            visibility: visible;
            }
            </style>
                    """, unsafe_allow_html=True)             
              
              
 # Select box for model type and output type      
  with h_col2:
   st.subheader('2. Run Prediction')
   st.write('Choose model and print the results')
   with st.expander("Click to print", expanded=False): 
    if st.session_state.reset_clicked:
       st.subheader("Enter the input values.")
       # Reset the flag so the message won't show again until next reset
       st.session_state.reset_clicked = False
    else:
    
     st.write('')
     # Finalize model input
  
     patient_aft = pd.DataFrame({'Age':Age_calculated, 'onset_site':onset_site_calculated, 'mean_R3_Respiratory_Insufficiency':mean_R3_Respiratory_Insufficiency, 'mean_weight':mean_weight, 
                                 'diag_delta':diag_delta_calculated, 'mean_Q5_Cutting':mean_Q5_Cutting, 'mean_creatinine':mean_creatinine, 'mean_Q2_Salivation':mean_Q2_Salivation, 
                                 'mean_fvc':mean_fvc, 'mean_Q3_Swallowing':mean_Q3_Swallowing, 'mean_Q7_Turning_in_Bed':mean_Q7_Turning_in_Bed, 
                                 'onset_delta':onset_delta_calculated, 'mean_bulbar':mean_bulbar}
                              , index = ['Your patient'])

     patient_cph = pd.DataFrame({'mean_bulbar':mean_bulbar, 'onset_delta':onset_delta_calculated, 'mean_Q3_Swallowing':mean_Q3_Swallowing, 'mean_fvc':mean_fvc, 
                                 'mean_Q7_Turning_in_Bed':mean_Q7_Turning_in_Bed, 'mean_Q2_Salivation':mean_Q2_Salivation, 'mean_creatinine':mean_creatinine, 'mean_weight':mean_weight, 
                                 'mean_ALSFRS_R_Total':mean_ALSFRS_R_Total, 'diag_delta':diag_delta_calculated, 'mean_Q5_Cutting':mean_Q5_Cutting, 'mean_R3_Respiratory_Insufficiency':mean_R3_Respiratory_Insufficiency, 
                                 'onset_site':onset_site_calculated, 'Age':Age_calculated}
                               , index = ['Your patient'])
   
     patient_rsf = pd.DataFrame({'mean_Q2_Salivation':mean_Q2_Salivation, 'Age':Age_calculated, 'mean_fvc':mean_fvc, 'mean_ALSFRS_R_Total':mean_ALSFRS_R_Total, 
                                'mean_Q3_Swallowing':mean_Q3_Swallowing, 'onset_delta':onset_delta_calculated, 'mean_bulbar':mean_bulbar, 'mean_Q1_Speech':mean_Q1_Speech}
                              , index = ['Your patient'])

     # Scale the patient info
     patient_aft_scaled_features = patient_aft[aft_scale_list]
     patient_aft_scaled_values = aft_scaler.transform(patient_aft_scaled_features)
     patient_aft[aft_scale_list] = patient_aft_scaled_values
    
     patient_cph_scaled_features = patient_cph[cph_scale_list]
     patient_cph_scaled_values = cph_scaler.transform(patient_cph_scaled_features)
     patient_cph[cph_scale_list] = patient_cph_scaled_values 



            
     result_rsf_o = rsf.predict_survival_function(patient_rsf, return_array=True)
     result_rsf = pd.DataFrame({'Time in months': rsf.event_times_.tolist(), 'Event-free probability': result_rsf_o[0].tolist()})
     result_rsf_concat = pd.concat([result_rsf, train_rsf[['Slow', 'Intermediate', 'Rapid']]], axis=1)
    
     result_aft = aft.predict_survival_function(patient_aft).reset_index()
     result_aft_concat = pd.concat([result_aft, train_aft[['Slow', 'Intermediate', 'Rapid']]], axis=1)
    
     result_cph = cph.predict_survival_function(patient_cph).reset_index()
     result_cph_concat = pd.concat([result_cph, train_cph[['Slow', 'Intermediate', 'Rapid']]], axis=1)
   
       
     # Click output
     if mean_ALSFRS_R_Total == 0:
         st.subheader('Enter the input values.')
     else:   
         st.subheader('Prediction as of the latest ALSFRS record ('+str(max_record)+')')

         pltcol1, pltcol2 = st.columns([2.5, 1])  # Adjust the ratio as needed
    
           # Column 2: Plotting options
         with pltcol2:
               option = st.radio("Select model type", ["Random Survival Forests", "Cox Proportional Hazard", "Accelerated Failure Time"])
               st.write('')
               st.write("Add options")
               show_prog = st.checkbox( 'Progressor groups', value=True)
               show_50 = st.checkbox('Rate Your Risk level', value=False)
               model_name_dic = {"Accelerated Failure Time":[aft, patient_aft, train_aft], "Cox Proportional Hazard":[cph, patient_cph, train_cph], "Random Survival Forests":[rsf, patient_rsf, train_rsf]}
               selected_model = model_name_dic[option][0]
               selected_patient = model_name_dic[option][1]
               selected_train = model_name_dic[option][2]
         if option == "Random Survival Forests":
            model_color = '#7DBA40'
         if option == "Cox Proportional Hazard":
            model_color = '#EE7100'
         if option == 'Accelerated Failure Time':
            model_color = '#008086'
            
         if selected_model == rsf:
            result_rsf = rsf.predict_survival_function(patient_rsf, return_array=True)
            result = pd.DataFrame({'Time in months': rsf.event_times_.tolist(), 'Event-free probability': result_rsf[0].tolist()})
            
         else:
            result = selected_model.predict_survival_function(selected_patient).reset_index()
               
         result.columns = ['Time in months', 'Event-free probability']
         result_concat = pd.concat([result, selected_train[['Slow', 'Intermediate', 'Rapid']]], axis=1)
         result_concat['Event-free probability'] = result_concat['Event-free probability'].round(4)
           # Column 1: Plot
         with pltcol1:
               empty_df = pd.DataFrame({'Time in months': [], 'Event-free probability': []})
               base_chart = alt.Chart(empty_df).encode(x='Time in months:Q', y='Event-free probability:Q').properties(width=600, height=400, color=alt.value('#000000'))
               layers = []
               
               line_Y_A = alt.Chart(result_concat).mark_line().encode(x='Time in months', y=alt.Y('Event-free probability', axis=alt.Axis(title='Event-free probability')), color=alt.value(model_color), strokeDash=alt.value([1])) # Sets the dash style of this line (solid)
               line_Y_B = alt.Chart(result_concat).mark_line().encode(x='Time in months', y='Slow', color=alt.value('gray'), strokeDash=alt.value([1, 5])) # Sets the dash style of this line (dashed)
               line_Y_C = alt.Chart(result_concat).mark_line().encode(x='Time in months', y='Intermediate', color=alt.value('gray'),strokeDash=alt.value([1, 5])) # Sets the dash style of this line (dot-dashed)
               line_Y_D = alt.Chart(result_concat).mark_line().encode(x='Time in months', y='Rapid', color=alt.value('gray'),strokeDash=alt.value([1, 5])) # Sets the dash style of this line (dot-dashed)
                   
               # Create text annotations for each line
               label_Y_A = alt.Chart(pd.DataFrame({'x': [result_concat['Time in months'].max()], 
                                                   'y': ([result_concat['Event-free probability'].iloc[-1]]), 
                                                   'text': ['Your patient']})).mark_text(align='left', dx=-200, dy=-8).encode(x='x:Q', 
                                                                                                                    y='y:Q',  
                                                                                                                    text='text:N')
    
               label_Y_B = alt.Chart(pd.DataFrame({'x': [result_concat['Time in months'].max()], 
                                                   'y': [result_concat['Slow'].iloc[-1]], 
                                                   'text': ['Slow']})).mark_text(align='left', dx=-10, dy=-6).encode(x='x:Q',
                                                                                                            y='y:Q',
                                                                                                            text='text:N')
    
               label_Y_C = alt.Chart(pd.DataFrame({'x': [result_concat['Time in months'].max()], 
                                                   'y': [result_concat['Intermediate'].iloc[-1]], 
                                                   'text': ['Intermediate']})).mark_text(align='left', dx=-50, dy=-6).encode(x='x:Q',
                                                                                                                   y='y:Q',
                                                                                                                   text='text:N')
    
               label_Y_D = alt.Chart(pd.DataFrame({'x': [result_concat['Time in months'].max()], 
                                                   'y': [result_concat['Rapid'].iloc[-1]], 
                                                   'text': ['Rapid']})).mark_text(align='left', dx=-15, dy=-6).encode(x='x:Q',
                                                                                                            y='y:Q',
                                                                                                            text='text:N')
               if selected_model == aft:
                  patient_median = aft.predict_percentile(patient_aft, ancillary=None, p=0.5)[0]
                  percentile = find_percentile(patient_median, np.array(train_median['AFT']))
                      
               elif selected_model == cph:
                  patient_median_orig = cph.predict_percentile(patient_cph, p=0.5)
                  if patient_median_orig == np.inf:
                      patient_median = (0.5*(result_concat['Time in months'].iloc[-1]))/(1-(result_concat['Event-free probability'].iloc[-1]))
                  else:
                      patient_median = patient_median_orig
                      
                  percentile = find_percentile(patient_median, np.array(train_median['COX']))
                  
               else:
                  patient_median_orig = predict_rsf_percentile(patient_rsf, 0.5)
                  if patient_median_orig == np.inf:
                      patient_median = (0.5*(result_concat['Time in months'].iloc[-1]))/(1-(result_concat['Event-free probability'].iloc[-1]))
                  else:
                      patient_median = patient_median_orig           
                  percentile = find_percentile(patient_median, np.array(train_median['RSF']))   
               
               # Find the point in 'result' where 'Time in months' is closest to patient_median
               closest_point_df = result_concat.iloc[(result_concat['Time in months'] - patient_median).abs().argsort()[:1]]
               
               
               # Create the red spot
               red_spot = alt.Chart(closest_point_df).mark_point(color='red').encode(
                    x='Time in months',
                    y='Event-free probability'
                )
            
               # Create the text annotation
               text_annotation = alt.Chart(closest_point_df).mark_text(
                    align='left',
                    baseline='middle',
                    dx=7  # Adjust for the right side
                ).encode(
                    x='Time in months',
                    y='Event-free probability',
                    text=alt.value(f"p = 50% at {patient_median:.2f} months (Ranked at {ordinal(int(percentile))} percentile)")
                )
            
               
               # Combine the charts with labels   
               if show_prog:
                   if show_50:
                       layers.extend([line_Y_A, label_Y_A, line_Y_B, label_Y_B, line_Y_C, label_Y_C, line_Y_D, label_Y_D, red_spot, text_annotation])
                   else:
                       layers.extend([line_Y_A, label_Y_A, line_Y_B, label_Y_B, line_Y_C, label_Y_C, line_Y_D, label_Y_D])

               else:
                   if show_50:
                       layers.extend([line_Y_A, label_Y_A, red_spot, text_annotation])
                   else:
                       layers.extend([line_Y_A, label_Y_A])

               # Combine final charts
               final_chart = alt.layer(*layers)
               st.altair_chart(final_chart, use_container_width=True)
         
         st.write("""
                  **※ Progressor groups**
                  <br>
                  "Intermediate" is the interquartile range (IQR) of PRO-ACT data, stratified by the "time to 50% event probablity"
                  """, unsafe_allow_html=True)
         st.write("""
                  **※ Rate Your Risk level**
                  <br>
                  Individual ranking within the PRO-ACT data, stratified by the "time to 50% event probability"
                  <br>
                  Lower percentile ranking = Higher risk of event
                  """, unsafe_allow_html=True)
         
         # Explain about the prediction results
         st.subheader('(1) The Probability of Loss of Autonomy in Swallowing Function')
         p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
         
         with p_col1:
               sixmo = round(100 -float(result.loc[(result['Time in months'] - 6).abs().idxmin()]['Event-free probability'])*100, 1)
               p_col1.metric("~ 6 months", f"{sixmo}"+"%")
         
         with p_col2:
               twelmo = round(100 - float(result.loc[(result['Time in months'] - 12).abs().idxmin()]['Event-free probability'])*100, 1)
               btw6_12 = round(twelmo - sixmo, 1)
               p_col2.metric("6 ~ 12 months", f"{btw6_12}"+"%")
               
         with p_col3:
               eighmo = round(100-float(result.loc[(result['Time in months'] - 18).abs().idxmin()]['Event-free probability'])*100, 1)
               btw12_18 = round(twelmo - sixmo, 1)
               p_col3.metric("12 ~ 18 months", f"{btw12_18}"+"%")
               
         with p_col4:
               
               twofomo = round(100-float(result.loc[(result['Time in months'] - 24).abs().idxmin()]['Event-free probability'])*100, 1)
               btw18_24 = round(twofomo - eighmo, 1)
               p_col4.metric("18 ~ 24 months", f"{btw18_24}"+"%")
               
               foeighmo = round(100 -float(result.loc[(result['Time in months'] - 48).abs().idxmin()]['Event-free probability'])*100, 1)
         with p_col5:
               over_24 = round(100 - twofomo, 1)
               p_col5.metric("24 ~ months", f"{over_24}"+"%")
         
          
         # Apply the CSS hack to dashboard
         st.markdown("""
                    <style>
                    /* metric styling */
                    div [data-testid="metric-container"]{
			background-color: #EDF1F9;
			border: 2px solid white;
			padding: 5% 5% 5% 10%;
			border-radius: 7px;
			color: black;
			overflow-wrap: break-word;
            visibility: visible;
            }
            </style>
                    """, unsafe_allow_html=True)
         
            
    
         st.subheader('')
         st.subheader('(2) Feature Impact')
           
         hr_values = {'mean_bulbar':-0.8405, 
                      'onset_delta':-0.4234, 
                      'mean_Q3_Swallowing':-0.2105, 
                      'mean_fvc':-0.3547, 
                      'mean_Q7_Turning_in_Bed':-0.1463, 
                      'mean_Q2_Salivation':-0.1358, 
                      'mean_creatinine':-0.1213, 
                      'mean_weight':-0.0986, 
                      'mean_ALSFRS_R_Total':-0.0, 
                      'diag_delta':-0.0671, 
                      'mean_Q5_Cutting':-0.0671, 
                      'mean_R3_Respiratory_Insufficiency':0.0655, 
                      'onset_site':0.0657, 
                      'Age':0.2802}
           
         average_features = {'mean_bulbar':-2.493006e-01, 
                      'onset_delta':1.964895e-01, 
                      'mean_Q3_Swallowing':-4.398557e-01, 
                      'mean_fvc':5.480296e-02, 
                      'mean_Q7_Turning_in_Bed':1.678772e-02, 
                      'mean_Q2_Salivation':-4.766441e-01, 
                      'mean_creatinine':5.551051e-02, 
                      'mean_weight':9.315467e-02, 
                      'mean_ALSFRS_R_Total':-8.453540e-02, 
                      'diag_delta':3.120188e-01, 
                      'mean_Q5_Cutting':-2.060012e-01, 
                      'mean_R3_Respiratory_Insufficiency':-6.688751e-02, 
                      'onset_site':0.180212, 
                      'Age':-8.529643e-02}
           
           
         # Function to calculate risk contribution for each feature
         def calculate_risk_contributions(hr_dict, patient_data):
               contributions = {}
               for feature, hr in hr_dict.items():
                   feature_value = float(patient_data[feature][0])-float(average_features[feature])
                   contribution = float(feature_value) * float(hr)
                   contributions[feature] = contribution
               return contributions
    
         # Calculate risk contributions for a specific patient
         patient_contributions = calculate_risk_contributions(hr_values, patient_cph)
    
         # Normalize contributions for visualization
         sorted_contributions = dict(sorted(patient_contributions.items(), key=lambda item: (item[1]), reverse=True))
         updated_contributions = {f"{patient_cph.iloc[0][key]} = {key}": value for key, value in sorted_contributions.items()}
           
         contributions_df = pd.DataFrame(list(sorted_contributions.items()), columns=['Feature', 'Contribution'])
    
         # Plot
         fig1=plt.figure(figsize=(8, 5))
         splot = sns.barplot(data=contributions_df, x='Contribution', y='Feature', orient='h', palette='RdYlBu')
         sns.set_style("whitegrid", {'grid.linestyle': ':'})
         plt.axvline(x=0, color='black', ls='-', lw=1.0)
         plt.ylabel('')
         plt.xlabel('Feature Impact')
         # get x-axis limits of the plot
         xabs_max = abs(max(splot.get_xlim(), key=abs))
         plt.xlim([-xabs_max-0.5, xabs_max+0.5])
         splot.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
         plt.bar_label(splot.containers[0], size=8, label_type='edge', fmt="%.4f") #fmt stands for 4th decimal
         plt.title(' Risk-decreasing                                          Risk-increasing ')
           
         if selected_model == cph:
           st.write("""
                  **※ Definition of Feature Impact**
                  <br>
                  = [Feature HR (Hazard ratio) ] × [Your Feature value input - Average Feature value in PRO-ACT data]
                  """, unsafe_allow_html=True)
           st.write('To find the definition of the names of feature variables, please refer to the original study paper.')       
           st.pyplot(fig1)
           
         else:
          
          st.write("""
                   please select
                   **Cox Proportional Hazard**
                   model to see the Feature impact analysis.
                   """)
        

with tab2:
  # About을 누르면 표시될 내용
  
  # Define the data
  data = [
    ['0.851 (±0.014)', '0.803 (±0.039)', "p = 0.999 (Accurate if p ≥ 0.05)", "p = 0.275"],
    ['0.850 (±0.014)', '0.802 (±0.040)', "p = 0.999", "p = 0.272"],
    ['0.846 (±0.013)', '0.785 (±0.044)', "p = 0.999", "p = 0.792"]
     ]

  # Define the multi-level column structure
  columns = pd.MultiIndex.from_tuples([
        ('C - index', 'Int. validation'),
        ('C - index', 'Ext. validation'),
        ('D-calibration', 'Int. validation'),
        ('D-calibration', 'Ext. validation'),
    ])
    
  # Create the DataFrame
  performance = pd.DataFrame(data, 
                      index=["AFT (Accelerated Failure Time)", 
                             "COX (Cox Proportional Hazard)", 
                             "RSF (Random Survival Forests)"],
                      columns=columns)
  
  st.subheader('Disclaimer: For Research Purposes Only')
  st.write("""
                  This web application is developed for research purposes. 
                  <br>
                  We do not assume responsibility for any medical practice or patient care decisions made based on the information provided by this application, 
                  <br>
                  and users should be aware that the application is exercised at their own risk. 
                  """, unsafe_allow_html=True)
  
  
  
  
  
  st.write('')
  st.subheader('Model Description')
  st.write("""
                  **※ Target Variable**
                  <br>
                  The model predicts the time to loss of autonomy in swallowing function for ALS patient, defined as when ALSFRS swallowing item score decreases to 1 or less, signaling the need for supplemental tube feeding.
                  
                  """, unsafe_allow_html=True)
  st.write("""
                  **※ Data and Vaildation**
                  <br>
                  The models are internally and externally validated to enhance the model's robustness and relevance in practical settings. 
                  <br>
                  From the PRO-ACT database, which contains records of 11,675 patients, we selected 3,396 patients for training and internal validation. 
                  <br>
                  The models were further validated externally using data with 207 patients from the Seoul National University Hospital ALS/MND Registry.
                  """, unsafe_allow_html=True)          
  
  
  st.write("""
                  **※ Model Performance**
                  """, unsafe_allow_html=True)   
  perf_col1, perf_col2 = st.columns([2,1])
  with perf_col1:
   st.table(performance)
  st.write('For details about study methodology and findings, please refer to the original research paper.')
  st.write('')
  st.subheader('How To Use')
  
  
  st.write("""    
                  **1. Access the Prediction Tool**
                  <br>
                  Begin by navigating to the "Run Prediction" tab on our web application.
                  """, unsafe_allow_html=True)  
  
    
  st.write("""    
                  **2. Enter Patient Information**
                  <br>
                  Input relevant patient data, including demographics, ALS history, FVC, Creatinine, Weight and ALSFRS-R scores. 
                  <br>
                  Ensure that the information is accurate and complete for the most reliable prediction.
                  <br>
                  For unknown values, try using the default value which is the average in PRO-ACT data.
                  """, unsafe_allow_html=True)   
                  
  st.write("""    
                  **3. Submit for Prediction**
                  <br>
                  Go to "Click to Save" button and you will find messages for a successful input save.
                  <br>
                  Expand the result section by "Click to print".
                  <br>
                  Click "Reset" if you want to start over.
                  """, unsafe_allow_html=True)                     
  
  st.write("""    
                  **4. View Prediction Results**
                  <br>
                  The prediction results will be displayed on the right side of your screen.
                  <br>
                  A plot of the Event-free probability is provided. Select options with model types by using the radio buttons, and review the comparative analysis by using the checkboxes.
                  <br>
                  Below that "(1) The probability of loss of autonomy in swallowing function" serves you the event probability in specific time ranges.
                  <br>
                  Below that "(2) Feature impact" which is available in the Cox proportional hazard model, provides the individual's feature impact based on the Hazard Ratios and feature value input.
                  """, unsafe_allow_html=True)           
  
  st.write('')
  st.subheader('The PRO-ACT Database')
  st.write("""
                  The Pooled Resource Open-Access ALS Clinical Trials (PRO-ACT) database is a cornerstone of our predictive model. 
                  <br>
                  It is one of the largest publicly available databases for ALS clinical trials, encompassing de-identified patient data from numerous completed ALS clinical studies.
                  """, unsafe_allow_html=True)
  st.write("""    
                  **※ Data Access**
                  <br>
                  The PRO-ACT data was accessed at https://nctu.partners.org/proact in October 2023 which has not been updated thereafter as of August 2022.  
                  """, unsafe_allow_html=True)
  st.write("""    
                  **※ Our Purpose of Use**
                  <br>
                  **(1) Model Training and Validation**
                  <br>
                  We utilized data from 3,396 patients in the PRO-ACT database to train and internally validate our predictive models. 
                  <br>
                  This large dataset ensures that our models are robust and reliable.
                  """, unsafe_allow_html=True)
  st.write("""    
                  **(2) Comparative Analysis Enabled For Individual Results**
                  <br>
                  In our web application, comparative framework by stratification of progressor group offers users a more comprehensive understanding of an individual patient's condition in the context of broader progression categories. 
                  <br>
                  Our models categorize patients in the PRO-ACT data into three distinct progression categories: Slow, Intermediate, and Rapid progressors. 
                  <br>
                  This stratification is based on their individual prediction made by the models.
                  """, unsafe_allow_html=True)
  
    
  
  st.write('')  
  st.subheader('Contact Us')
  st.markdown('''If you encounter any issues or have questions, please contact our support team.
              (nrhong@gmail.com)''')

  st.markdown("""
  <style>
  #the-title {
  text-align: center
  }
  </style>
  """, unsafe_allow_html=True)  
