import streamlit as st
from PIL import Image
import pandas as pd
import csv
import re
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer

from sklearn.model_selection import train_test_split
import math
import sklearn.metrics
from sklearn.metrics import r2_score

st.set_page_config(page_title='Used Car Price Prediction', layout='wide')

image = Image.open('car.jpg')

st.image(image)

st.title("How Much Is Your Car Worth?")

#Load Data
car_df= pd.read_csv('./car_df_for_modelling.csv')
car_df= car_df.drop('Unnamed: 0', axis=1)

brand_list= car_df['car_brand'].unique().tolist()
brand_list.sort()
print(brand_list)

#brand = None
#COE_cat = None
#OMV = None
#ARF = None
#road_tax = None
#COE_left = None
#power = None
#age = None
#mileage = None

col1 = st.columns(1)[0]
container1 = st.container()
brand = container1.selectbox('Car Brand', brand_list)

with container1:
    col1, col2, col3 = st.columns(3)

    COE_cat_list = ['Category A', 'Category B']
    COE_cat= col1.selectbox('COE Category', COE_cat_list)

    road_tax= col2.number_input('Road Tax Paid Per Year (SGD)')
    
    dereg_value= col3.number_input('Deregistration Value as of Today (SGD)')

container2 = st.container()

with container2:
    col1, col2= st.columns(2)
            
    OMV= col1.number_input('Open Market Value (SGD)')

    ARF= col2.number_input('Additional Registration Fee (SGD)')
    
container3 = st.container()

with container3:
    col1, col2= st.columns(2)

    COE_left= col1.number_input('Total COE Left (Number of Days)')

    power= col2.number_input('Power (kW)')
    
container4 = st.container()

with container4:
    col1, col2= st.columns(2)

    age= col1.number_input('Age of Car (Number of Years)')

    mileage= col2.number_input('Total Mileage (kilometers)')
    
submitted = st.button('Submit')


luxury_brand_list= ['Mercedes-Benz', 'BMW', 'Audi', 'Porsche', 'Land-Rover', 'Jaguar', 'Lexus', 'Volvo', 'Rolls-Royce', 
                    'Infiniti', 'Maserati', 'Bentley', 'Maybach', 'Lamborghini', 'Alfa', 'DS7', 'Daimler', 'Tesla', 
                    'CUPRA', 'MINI']

if brand in luxury_brand_list:
    brand_cat = 'luxury'
else:
    brand_cat = 'regular'
    
if brand_cat == 'luxury' and COE_cat == 'Category B':
    car_cat = 4
elif brand_cat == 'normal' and COE_cat == 'Category B':
    car_cat = 3
elif brand_cat == 'luxury' and COE_cat == 'Category A':
    car_cat = 2
else:
    car_cat = 1 #normal car & Cat A
    

selected_features= ['dereg_value', 'OMV', 'ARF', 'road_tax_per_yr', 'total_mileage', 'Total_COE_left', 'Power_kw', 'car_age', 
                     'car_category']

X = car_df[selected_features]
y = car_df.Price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state= 42)

params = {'objective': 'reg:squarederror', 'n_estimators': 100, 'max_depth': 9}
model = xgb.XGBRegressor(**params)

scaler = QuantileTransformer(output_distribution="normal")
X_train = scaler.fit_transform(X_train)

# fit the train data set to the model:
model.fit(X_train, y_train)

user_input= [[dereg_value, OMV, ARF, road_tax, mileage, COE_left, power, age, car_cat]]
user_input= scaler.transform(user_input)

y_pred= model.predict(user_input)
predicted_price= round(y_pred[0], 0)

st.subheader('Predicted Price')

if submitted:
    st.metric('Predicted Price', f"${predicted_price}")
    
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
    
image2 = Image.open('girl.jpg')
st.image(image2)

st.write(f"A machine learning model for used car price prediction created by Er Ning")
st.write("Check out my [LinkedIn](https://www.linkedin.com/in/er-ning-582a3b12a/) and [GitHub](https://github.com/NingDS)")

