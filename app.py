import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Title
st.title("🏠 House Price Prediction App")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("housing.csv")
    return df

df = load_data()

# Train the model
model = LinearRegression()
X = df[["area", "bedrooms", "bathrooms"]]
y = df["price"]
model.fit(X, y)

# Sidebar input
st.sidebar.header("Enter House Details")

area = st.sidebar.number_input("Area (in sqft)", min_value=500, max_value=10000, value=1000)
bedrooms = st.sidebar.slider("Bedrooms", 1, 6, 2)
bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 1)

# Prediction
input_data = np.array([[area, bedrooms, bathrooms]])
predicted_price = model.predict(input_data)[0]

st.subheader("📊 Predicted Price")
st.success(f"Estimated price: ₹ {predicted_price:.2f} lakhs")

# Optional: Show dataset
if st.checkbox("Show Training Data"):
    st.dataframe(df)

# Optional: Plot
if st.checkbox("Show Price vs Area Plot"):
    plt.scatter(df["area"], df["price"], color='blue')
    plt.xlabel("Area (sqft)")
    plt.ylabel("Price (₹ in lakhs)")
    st.pyplot(plt)
