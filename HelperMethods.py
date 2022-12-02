import joblib
import pandas as pd 
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