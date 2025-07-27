import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib


@st.cache_data
def load_data():
    return pd.read_csv("car_data.csv")

df = load_data()


df['Car_Age'] = 2025 - df['Year']
df.drop(['Year', 'Car_Name'], axis=1, inplace=True)

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

cat_cols = ['Fuel_Type', 'Seller_Type', 'Transmission']
num_cols = ['Present_Price', 'Kms_Driven', 'Owner', 'Car_Age']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), cat_cols)
    ],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('pre', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

joblib.dump(model, "car_price_model.pkl")

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

st.title("🚗 Car Selling Price Prediction")
st.write("Estimate the selling price of your car based on its condition and specifications.")

present_price = st.slider("Showroom Price (in Lakhs)", 0.0, 50.0, 5.0, 0.1)
kms_driven = st.number_input("Kilometers Driven", 0, 500000, 30000)
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
car_age = st.slider("Car Age (Years)", 0, 30, 5)

if st.button("Predict Price"):
    input_df = pd.DataFrame({
        'Present_Price': [present_price],
        'Kms_Driven': [kms_driven],
        'Owner': [owner],
        'Fuel_Type': [fuel_type],
        'Seller_Type': [seller_type],
        'Transmission': [transmission],
        'Car_Age': [car_age]
    })

    model = joblib.load("car_price_model.pkl")
    pred_price = model.predict(input_df)[0]
    st.success(f"💰 Estimated Selling Price: ₹ {pred_price:.2f} Lakhs")

st.write(f"📊 Model Accuracy (R² Score): {r2:.2f}")

if st.checkbox("Show Feature Importance"):
    onehot = model.named_steps['pre'].named_transformers_['cat']
    onehot_features = onehot.get_feature_names_out(cat_cols)
    feature_names = np.concatenate([onehot_features, num_cols])

    regressor = model.named_steps['regressor']
    importances = regressor.feature_importances_

    fig, ax = plt.subplots()
    ax.barh(feature_names, importances)
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance Score")
    ax.invert_yaxis()
    st.pyplot(fig)
