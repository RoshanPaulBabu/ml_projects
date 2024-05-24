import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load the dataset
dataset = pd.read_csv('salary.csv')

# Extracting the independent and dependent variables
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Building the Polynomial regression model
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Prediction model
st.title("Salary Prediction")

position_level = st.number_input("Enter Position Job Level:", min_value=int(x.min()), max_value=int(x.max()), value=int(x.min()), step=1)
prediction_poly = lin_reg_2.predict(poly_reg.fit_transform([[position_level]]))

# Display predicted salary with two decimal places
predicted_salary = round(prediction_poly[0], 2)
st.write(f"Predicted Salary (Polynomial Regression): ${predicted_salary}")
