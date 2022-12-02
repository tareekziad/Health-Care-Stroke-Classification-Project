import pandas as pd 
import warnings
import streamlit as st
import joblib
from sklearn.preprocessing import FunctionTransformer
import sklearn
warnings.filterwarnings('ignore')

#----------------------------------------------------------------------------------------------------------------

def bmi_glu_preprocessing(x):
    bmi_bins , bmi_labels , bmi_mapper = joblib.load('bmi_bins.h5') , joblib.load('bmi_labels.h5') , joblib.load('bmi_mapper.h5')
    glucose_bins , glucose_labels = joblib.load('glucose_bins.h5') , joblib.load('glucose_labels.h5')
    glucose_mapper = joblib.load('glucose_mapper.h5')
    x['glucose_cat'] = pd.cut(x['avg_glucose_level'], bins = glucose_bins, labels = glucose_labels)
    x['bmi_cat'] = pd.cut(x['bmi'], bins = bmi_bins, labels = bmi_labels)
    x['glucose_cat'] = x['glucose_cat'].map(glucose_mapper).astype(int)
    x['bmi_cat'] = x['bmi_cat'].map(bmi_mapper).astype(int)
    x.drop(['bmi' , 'avg_glucose_level'] , axis = 1 , inplace = True )
    return x

#----------------------------------------------------------------------------------------------------------------

SVC = joblib.load('SVC.h5')
SVC.steps.insert(0 , ('preprocessing' , FunctionTransformer(bmi_glu_preprocessing)))

#----------------------------------------------------------------------------------------------------------------

def Predict(gender, age, hypertension, heart_disease, ever_married, work_type,Residence_type,avg_glucose_level,bmi,smoking_status):
    test = pd.DataFrame(columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status'])
    
    test.at[0,['gender']] = gender
    test.at[0,['age']] = age
    test.at[0,['hypertension']] = hypertension
    test.at[0,['heart_disease']] = heart_disease
    test.at[0,['ever_married']] = ever_married
    test.at[0,['work_type']] = work_type
    test.at[0,['Residence_type']] = Residence_type
    test.at[0,['avg_glucose_level']] = avg_glucose_level
    test.at[0,['bmi']] = bmi
    test.at[0,['smoking_status']] = smoking_status
        
    return SVC.predict(test)[0]
    
#----------------------------------------------------------------------------------------------------------------
    
    
def main():
    st.header('estimate your stroke'.capitalize())
    st.image('images.jfif')
    gender = st.selectbox('what is your gender'.title() , ['Male', 'Female',])
    age = st.slider('what is your age'.title() , min_value = 1 , max_value=85 , value=20 , step=1)
    hypertension = st.selectbox('have you hypertension'.title() , ['NO', 'YES'])
    heart_disease = st.selectbox('have you heart disease'.title() , ['YES', 'NO'])
    ever_married = st.selectbox('have you ever married'.title() , ['Yes', 'No'])
    work_type = st.selectbox('what is your work type'.title() , ['Private', 'Self-employed', 'Govt_job', 'Never_worked'])
    Residence_type = st.selectbox('what is your Residence type'.title() , ['Urban', 'Rural'])
    avg_glucose_level = st.slider('enter youe glucose ratio'.title() ,min_value = 50.0 , max_value=280.0 , value=70.0 , step=0.1)
    bmi = st.slider('enter your bmi'.title() ,min_value = 10.0 , max_value=100.0 , value=70.0 , step=0.2 )
    smoking_status = st.selectbox('have you smoke'.title() , ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])
    
    if st.button('predict'.title()):
        ans = Predict(gender, age, hypertension, heart_disease, ever_married,
                      work_type,Residence_type,avg_glucose_level,bmi,smoking_status)
        if ans:
            st.write('Unfortunately, you are prone to a stroke, you should consult a doctor'.title() , ans)
        else :
            st.write('No need to worry, you are in good health'.title() , ans)
main()
