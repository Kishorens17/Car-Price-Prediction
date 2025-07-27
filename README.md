# 🚗 Car Selling Price Prediction System

This project predicts the **resale value of a car** using a machine learning model built with Python and Streamlit. It allows users to interactively enter car attributes and receive a highly accurate price prediction, along with insight into which factors influence the price the most.

---

## 📊 Project Overview

The app provides users an easy way to evaluate their car’s market value using machine learning. It considers various car features and utilizes a **Random Forest Regressor** for prediction.

### 🎯 Features:
- Streamlit-based web interface
- Random Forest regression model
- Feature engineering (Car Age)
- Visualized feature importance
- Real-time prediction with high R² accuracy

---



## 📁 Dataset Format

Your CSV file (`car_data.csv`) should include:

```csv
Car_Name,Year,Selling_Price,Present_Price,Kms_Driven,Fuel_Type,Seller_Type,Transmission,Owner
ritz,2014,3.35,5.59,27000,Petrol,Dealer,Manual,0
```

## 💡 Benefits of this Work

- **Helps individual car sellers** determine a fair selling price without relying solely on dealerships.
- **Assists used car dealers** in rapidly assessing the value of vehicles in inventory.
- **Saves time** by providing instant price estimation without manual evaluation.
- **Data-driven insights**: Users learn what features most influence the value of their car.
- Can be extended into a **full-scale pricing API** for online car marketplaces.

---
