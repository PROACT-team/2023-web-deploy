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

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import altair as alt 

import lifelines
import sksurv.ensemble

import pickle
import cloudpickle as cp
from urllib.request import Request, urlopen
import zipfile
import os


# when using repository files...
train_aft = pd.read_csv('data/train_aft_1109.csv')
train_cph = pd.read_csv('data/train_cph_1109.csv')
train_rsf = pd.read_csv('data/train_rsf_1109.csv')
train_median = pd.read_csv('data/train_median_1109.csv')

def extract_and_load_pkl(zip_path, pkl_filename, extract_to='.'):
    # Extract the pkl file from the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract(pkl_filename, extract_to)

    pkl_path = os.path.join(extract_to, pkl_filename)

    # Load the extracted pkl file
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)

    # Optionally, remove the extracted file if you don't need it
    os.remove(pkl_path)

    return data

aft = pickle.load(open('data/aft_1109.pkl', 'rb'))
cph = pickle.load(open('data/cph_1109.pkl', 'rb'))
rsf = extract_and_load_pkl('data/rsf_1109.zip', 'rsf_1109.pkl')


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


# page layout
st.set_page_config(layout='wide')

# Set title
ti_col1, ti_col2 = st.columns([3.15,0.76])
with ti_col1:
    st.title('Predicting Loss of Autonomy in Swallowing Function in ALS Patients')
    st.subheader('Web application ver.')
    st.subheader('')
with ti_col2:
    st.image('https://github.com/PROACT-team/2022-Final-scripts/assets/78291206/3c6b0278-157a-4185-b4dd-9fa90690fe58')

# Create Tabs (page shifting)
tab1, tab2, tab3, tab4 = st.tabs(['Home' , 'How to use', 'Run Prediction', 'Contact us'])

# Tab style
st.markdown("""
<style>

	.stTabs [data-baseweb="tab-list"] {
		gap: 5px;
        font-size: 1rem;
    }

	.stTabs [data-baseweb="tab"] {
		height: 50px;
        white-space: pre-wrap;
		background-color: #F9F9F9;
        font-size: 1rem;
        margin:0;
        width: 100%;
		border-radius: 2px 2px 0px 0px;
		gap: 7px;
		padding-top: 10px;
		padding-bottom: 10px;
    }

	.stTabs [aria-selected="true"] {
  		background-color: None;
	}

</style>       
""", unsafe_allow_html=True)






with tab1:
  # Home을 누르면 표시될 내용
  st.subheader('Welcome!')
  st.write('This webpage will provide you an individualized prediction for loss of autonomy in swallowing function in ALS patients.')
  st.write('')
  st.write('')
  st.write('')
  st.write('')
  st.write('')

  st.markdown("""
  <style>
  #the-title {
  text-align: center
  }
  </style>
  """, unsafe_allow_html=True)    





with tab2:
  # How to use를 누르면 표시될 내용 
  st.subheader('Describe instructions here..')
  st.subheader('... ')
  st.subheader('Model Performance Description')
  st.subheader('... ')
  st.subheader('About training dataset')
  st.subheader(' ')






with tab3:
  # Run prediction을 누르면 표시될 내용 
 st.title('')
 h_col1, h_col2= st.columns([5.2,6])
 with h_col1:
  st.subheader('1. Patient Info') 
  st.write('Please fill in the input values. Use default values for unknown data') 
  with st.expander("Patient Info", expanded=True):   
  # Patient feature input box
   with st.form(key='input_box'):
      # Static features
      st.write('Use default values for unknown data')
      st.subheader('Demographics & ALS history')     
      col1, col2, = st.columns(2)
      with col1:
          Age = st.text_input(label='Age at diagnosis (years)', value=56)
          onset_site = st.selectbox('Onset region', ['Non-Bulbar', 'Bulbar'])
      with col2:
          onset_delta = st.text_input(label='Time from symptom onset (months)', value=19)
          diag_delta = st.text_input(label='Time from diagnosis (months)', value=8)
          
      # Time-resolved features
      st.subheader('Features over the past 3 months')
      t_col1, t_col2 = st.columns([1.5, 2.5])
      with t_col1:
          mean_ALSFRS_R_Total = st.text_input('Mean ALSFRS-R total score (points)', value=40)
          mean_bulbar = st.text_input('Mean ALSFRS-R bulbar (Q1+Q3) score (points)',value=8)
          mean_Q1_Speech = st.text_input('Mean ALSFRS-R Q1_Speech (points)',value=3.2)
          mean_Q2_Salivation = st.text_input('Mean ALSFRS-R Q2_Salivation (points)', value=3.4)
          mean_Q3_Swallowing = st.text_input('Mean ALSFRS-R Q3_Swallowing (points)',  value=3.6)
          mean_Q5_Cutting = st.text_input('Mean ALSFRS-R Q5_Cutting (points)', value=2.6)
          mean_Q7_Turning_in_Bed = st.text_input('Mean ALSFRS-R Q7_Turning_in_Bed (points)', value=3.0)
          mean_R3_Respiratory_Insufficiency = st.text_input('Mean ALSFRS-R R3_Respiratory_Insufficiency (points)', value=4.0)
          mean_fvc = st.slider('Mean Forced Vital Capacity (%)', min_value=10, max_value = 100, value=70, step=5)
          mean_Creatinine = st.slider('Mean Creatinine (umol/L)', min_value=10, max_value = 100, value=50, step=5)  
      
      with t_col2:
          slope_ALSFRS_R_Total = st.text_input('Slope of ALSFRS-R total score (points per month)   *slope can be negative', value=-0.83)
          slope_bulbar = st.text_input('Slope of ALSFRS-R bulbar (Q1+Q3) score (points per month)   *slope can be negative', value=-0.07)
          slope_Q1_Speech = st.text_input('Slope of ALSFRS-R Q1_Speeech (points per month)   *slope can be negative', value=-0.04)
          slope_Q2_Salivation = st.text_input('Slope of ALSFRS-R Q2_Salivation (points per month)   *slope can be negative', value=-0.04)
          slope_Q6_Dressing_and_Hygiene = st.text_input('Slope of ALSFRS-R Q6_Dressing_and_Hygiene (points per month)   *slope can be negative', value=-0.12)
          slope_Q8_Walking = st.text_input('Slope of ALSFRS-R Q8_Walking (points per month)   *slope can be negative', value=-0.07)
          slope_R3_Respiratory_Insufficiency = st.text_input('Slope of ALSFRS-R R3_Respiratory_Insufficiency (points per month)   *slope can be negative', value=-0.02)
          mean_weight = st.text_input('Mean weight (kg)', value=78)
          slope_weight = st.text_input('Weight change rate (kg per month)              *change rate can be negative', value=-0.2)
      st.form_submit_button('Click to Save')
      Age_calculated = (float(Age)-float(15))//5
      onset_delta_calculated = (float(onset_delta)-3.0)
      diag_delta_calculated = (float(diag_delta)-3.0)
      if onset_site == 'Bulbar':
          onset_site_calculated = 1
      else :
          onset_site_calculated = 0
      
  # Select box for model type and output type      
 with h_col2:
  st.subheader('2. Run Prediction')
  st.write('Choose model and print the results')
  with st.expander("Run Prediction", expanded=False):  
   option = st.selectbox('Select model type',['Random survival forests (Default)', 'Accelerated failure time', 'Cox proportional hazard'])
   st.write('Show options (reference curves):')
   check_progressor = st.checkbox('Show progressor curves')
   st.write(' ')
   
   # Finalize model input
   patient_aft = pd.DataFrame({'onset_site': onset_site_calculated, 'Age':Age_calculated, 'mean_R3_Respiratory_Insufficiency':mean_R3_Respiratory_Insufficiency, 'slope_ALSFRS_R_Total':slope_ALSFRS_R_Total,
                          'mean_Q7_Turning_in_Bed':mean_Q7_Turning_in_Bed, 'mean_Q2_Salivation':mean_Q2_Salivation, 'slope_Q2_Salivation':slope_Q2_Salivation,
                          'slope_R3_Respiratory_Insufficiency':slope_R3_Respiratory_Insufficiency, 'slope_Q8_Walking':slope_Q8_Walking, 'slope_Q1_Speech':slope_Q1_Speech,
                          'mean_Q3_Swallowing':mean_Q3_Swallowing, 'slope_Q6_Dressing_and_Hygiene':slope_Q6_Dressing_and_Hygiene, 'mean_bulbar':mean_bulbar}, index = ['Your patient'])

   patient_cph = pd.DataFrame({'mean_bulbar':mean_bulbar, 'slope_Q6_Dressing_and_Hygiene':slope_Q6_Dressing_and_Hygiene, 'mean_Q3_Swallowing':mean_Q3_Swallowing, 'slope_Q1_Speech':slope_Q1_Speech,
                          'slope_Q2_Salivation':slope_Q2_Salivation, 'slope_Q8_Walking':slope_Q8_Walking, 'mean_Q2_Salivation':mean_Q2_Salivation, 'slope_ALSFRS_R_Total':slope_ALSFRS_R_Total,
                          'mean_Q7_Turning_in_Bed':mean_Q7_Turning_in_Bed, 'mean_R3_Respiratory_Insufficiency':mean_R3_Respiratory_Insufficiency, 'Age':Age_calculated, 'onset_site':onset_site_calculated}, index = ['Your patient'])
   
   patient_rsf = pd.DataFrame({'slope_ALSFRS_R_Total':slope_ALSFRS_R_Total, 'mean_weight':mean_weight, 'diag_delta':diag_delta_calculated, 'Age':Age_calculated, 'onset_site':onset_site_calculated, 'mean_fvc':mean_fvc,
                          'mean_Q3_Swallowing':mean_Q3_Swallowing, 'mean_ALSFRS_R_Total':mean_ALSFRS_R_Total, 'mean_Q2_Salivation':mean_Q2_Salivation, 'onset_delta':onset_delta_calculated,
                          'mean_Q1_Speech':mean_Q1_Speech, 'mean_bulbar':mean_bulbar}, index = ['Your patient'])

   
   # Model name dictionary
   model_name_dic = {"Accelerated failure time":[aft, patient_aft, train_aft], "Cox proportional hazard":[cph, patient_cph, train_cph], "Random survival forests (Default)":[rsf, patient_rsf, train_rsf]}
   selected_model = model_name_dic[option][0]
   selected_patient = model_name_dic[option][1]
   selected_train = model_name_dic[option][2]
   
   # Create click button
   button = st.button('Click to Print')
   
       
   # Click output
   if button:    
       if selected_model == rsf:
           result_rsf = rsf.predict_survival_function(patient_rsf, return_array=True)
           result = pd.DataFrame({'Time in months': rsf.event_times_.tolist(), 'Event-free probability': result_rsf[0].tolist()})
       else:
           result = selected_model.predict_survival_function(selected_patient).reset_index()
           
       result.columns = ['Time in months', 'Event-free probability']
       result_concat = pd.concat([result, selected_train[['Slow', 'Intermediate', 'Rapid']]], axis=1)
           
       if check_progressor:
           # Create the individual line charts
           line_Y_A = alt.Chart(result_concat).mark_line().encode(x='Time in months', y=alt.Y('Event-free probability', axis=alt.Axis(title='Event-free probability')), color=alt.value('#4B77A7'), strokeDash=alt.value([1])) # Sets the dash style of this line (solid)
           line_Y_B = alt.Chart(result_concat).mark_line().encode(x='Time in months', y='Slow', color=alt.value('gray'), strokeDash=alt.value([1, 5])) # Sets the dash style of this line (dashed)
           line_Y_C = alt.Chart(result_concat).mark_line().encode(x='Time in months', y='Intermediate', color=alt.value('gray'),strokeDash=alt.value([1, 5])) # Sets the dash style of this line (dot-dashed)
           line_Y_D = alt.Chart(result_concat).mark_line().encode(x='Time in months', y='Rapid', color=alt.value('gray'),strokeDash=alt.value([1, 5])) # Sets the dash style of this line (dot-dashed)
           # Combine the charts
           chart = alt.layer(line_Y_A, line_Y_B, line_Y_C, line_Y_D)
               
       else:
           chart = alt.Chart(result).mark_line().encode(x='Time in months', y='Event-free probability' )
               
       st.altair_chart(chart, use_container_width=True)
    
     # Explain about the prediction results
       st.subheader('Prediction results')


       st.subheader('(1) The Probability of Loss of Autonomy in Swallowing would be ..')
       p_col1, p_col2, p_col3, p_col4 = st.columns(4)
       with p_col1:
           st.write('6 months')
           sixmo = round(100 -float(result.loc[(result['Time in months'] - 6).abs().idxmin()]['Event-free probability'])*100, 1)
           st.subheader(str(sixmo)+ '%')
           st.write('30 months')
           thirmo = round(100 -float(result.loc[(result['Time in months'] - 30).abs().idxmin()]['Event-free probability'])*100, 1)
           st.subheader(str(thirmo)+ '%')
           
       with p_col2:
           st.write('12 months')
           twelmo = round(100 - float(result.loc[(result['Time in months'] - 12).abs().idxmin()]['Event-free probability'])*100, 1)
           st.subheader(str(twelmo)+'%')
           st.write('36 months')
           thirsmo = round(100 -float(result.loc[(result['Time in months'] - 36).abs().idxmin()]['Event-free probability'])*100, 1)
           st.subheader(str(thirsmo)+ '%')
       with p_col3:
           st.write('18 months')
           eighmo = round(100-float(result.loc[(result['Time in months'] - 18).abs().idxmin()]['Event-free probability'])*100, 1)
           st.subheader(str(eighmo)+'%')
           st.write('42 months')
           fotwomo = round(100 -float(result.loc[(result['Time in months'] - 42).abs().idxmin()]['Event-free probability'])*100, 1)
           st.subheader(str(fotwomo)+ '%')
       with p_col4:
           st.write('24 months')
           twofomo = round(100-float(result.loc[(result['Time in months'] - 24).abs().idxmin()]['Event-free probability'])*100, 1)
           st.subheader(str(twofomo)+ '%')
           st.write('48 months')
           foeighmo = round(100 -float(result.loc[(result['Time in months'] - 48).abs().idxmin()]['Event-free probability'])*100, 1)
           st.subheader(str(foeighmo)+ '%')


       st.subheader('')
       st.subheader('(2) Compared to the training data, your patient is ..')
       def find_speed(value, array):
           # Calculate the percentile rank of the input value within the array
           percentile = np.sum(array < value) / len(array) * 100
    
           # Determine the speed category based on the percentile
           if percentile < 25:
               progression_category = 'Rapid'
           elif percentile >= 25 and percentile < 75:
               progression_category = 'Intermediate'
           else:
               progression_category = 'Slow'
               
           return progression_category
       
       def find_percentile(value, array):
           # Calculate the percentile rank of the input value within the array
           percentile = np.sum(array < value) / len(array) * 100
               
           return percentile
        
       if selected_model == aft:
           patient_median = aft.predict_percentile(patient_aft, ancillary=None, p=0.5)[0]
           prog = find_speed(patient_median, np.array(train_median['AFT']))
           percentile = find_percentile(patient_median, np.array(train_median['AFT']))
           
       elif selected_model == cph:
           patient_median_orig = cph.predict_percentile(patient_cph, p=0.5)
           if patient_median_orig == np.inf:
               patient_median = (0.5*(result['Time in months'].iloc[-1]))/(1-(result['Event-free probability'].iloc[-1]))
           else:
               patient_median = patient_median_orig
           prog = find_speed(patient_median, np.array(train_median['COX']))
           percentile = find_percentile(patient_median, np.array(train_median['COX']))
       else:
           patient_median_orig = predict_rsf_percentile(patient_rsf, 0.5)
           if patient_median_orig == np.inf:
               patient_median = (0.5*(result['Time in months'].iloc[-1]))/(1-(result['Event-free probability'].iloc[-1]))
           else:
               patient_median = patient_median_orig           
           prog = find_speed(patient_median, np.array(train_median['RSF']))
           percentile = find_percentile(patient_median, np.array(train_median['RSF']))

       
       pr_col1,pr_col2,pr_col3 = st.columns(3)
       with pr_col1:
           st.write('Progressor category :')
           st.subheader(str(prog))
       with pr_col2:
           st.write('Ranked at PRO-ACT data being around : ')
           st.subheader(str(round(percentile))+' percentile')
       st.write(' ')
       st.write('※ Definition of [Rapid] / [Intermediate] / [Slow] : [0p ~ 24p] / [25p ~ 75p] / [75p ~ 100p]')
       st.write('※ The percentile ranking is determined by the "time to 50% probablity"')       

       st.subheader('')
       st.subheader('(3) Feature impact')
       
       hr_values = {"onset_site": 0.3008, 
                    "Age": 0.1018, 
                    "mean_R3_Respiratory_Insufficiency": 0.0093,
                    "mean_Q2_Salivation": -0.0930,
                    "mean_Q7_Turning_in_Bed": -0.1253,
                    "slope_Q2_Salivation": -0.1319,
                    "slope_Q8_Walking": -0.1611,
                    "slope_ALSFRS_R_Total": -0.1664,
                    "mean_Q3_Swallowing": -0.2611,
                    "slope_Q6_Dressing_and_Hygiene": -0.2818,
                    "slope_Q1_Speech": -0.3529,
                    "mean_bulbar": -0.4038}
       
       average_features = {"onset_site": 0.191760, 
                    "Age": 7.6764, 
                    "mean_R3_Respiratory_Insufficiency": 3.9231,
                    "mean_Q2_Salivation": 3.5093,
                    "mean_Q7_Turning_in_Bed": 3.0505,
                    "slope_Q2_Salivation": -0.0389,
                    "slope_Q8_Walking": -0.0735,
                    "slope_ALSFRS_R_Total": -0.8283,
                    "mean_Q3_Swallowing": 3.5484,
                    "slope_Q6_Dressing_and_Hygiene": -0.1167,
                    "slope_Q1_Speech": -0.0474,
                    "mean_bulbar": 6.8058}
       
       patient_cph_orig = pd.DataFrame({'mean_bulbar':mean_bulbar, 'slope_Q6_Dressing_and_Hygiene':slope_Q6_Dressing_and_Hygiene, 'mean_Q3_Swallowing':mean_Q3_Swallowing, 'slope_Q1_Speech':slope_Q1_Speech,
                          'slope_Q2_Salivation':slope_Q2_Salivation, 'slope_Q8_Walking':slope_Q8_Walking, 'mean_Q2_Salivation':mean_Q2_Salivation, 'slope_ALSFRS_R_Total':slope_ALSFRS_R_Total,
                          'mean_Q7_Turning_in_Bed':mean_Q7_Turning_in_Bed, 'mean_R3_Respiratory_Insufficiency':mean_R3_Respiratory_Insufficiency, 'Age':Age, 'onset_site':onset_site}, index = ['Your patient'])
       
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
       updated_contributions = {f"{patient_cph_orig.iloc[0][key]} = {key}": value for key, value in sorted_contributions.items()}
       
       contributions_df = pd.DataFrame(list(updated_contributions.items()), columns=['Feature', 'Contribution'])

       # Plot
       fig1=plt.figure(figsize=(8, 5))
       splot = sns.barplot(data=contributions_df, x='Contribution', y='Feature', orient='h', palette='RdYlBu')
       sns.set_style("whitegrid", {'grid.linestyle': '--'}, )
       plt.axvline(x=0, color='black', ls='-', lw=1.0)
       plt.ylabel('')
       plt.xlabel('Feature Impact')
       # get x-axis limits of the plot
       xabs_max = abs(max(splot.get_xlim(), key=abs))
       plt.xlim([-xabs_max-0.5, xabs_max+0.5])
       splot.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
       plt.bar_label(splot.containers[0], size=8, label_type='edge', fmt="%.4f") #fmt stands for 4th decimal
       plt.title('(Risk-decreasing)                                        (Risk-increasing)')
       
       if selected_model == cph:
           st.write('※ Feature impact = [Feature HR (Hazard ratio)] * [Individual Feature value input - Average Feature value in training data]')
       
           st.pyplot(fig1)
       else:
           st.write('Please select '+ ' Cox Proportional Hazard model ' +' to see the Feature impact analysis.')
       
with tab4:
  # Contact us를 누르면 표시될 내용 
  st.subheader('If you have any general question, contact: lasfpredictionmodel@gmail.com')

