# 🚗 Car Price Prediction – Linear Regression from Scratch

This project builds a **linear regression model from scratch** (no scikit-learn!) to predict the **selling price of used cars** using the [CarDekho Vehicle Dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho). It's a great example of how machine learning can model real-world relationships using gradient descent and simple math.

---

## 📊 Dataset Description

The dataset includes:
- `Car_Name` – name of the car
- `Year` – year of purchase
- `Selling_Price` – price the car was sold for (**target variable**)
- `Present_Price` – current ex-showroom price
- `Kms_Driven` – distance driven
- `Fuel_Type` – Petrol/Diesel/CNG
- `Seller_Type` – Dealer/Individual
- `Transmission` – Manual/Automatic
- `Owner` – Number of previous owners

---

## 📌 What’s Covered

- ✅ Manual **gradient descent** implementation
- ✅ Used only **NumPy** and **Pandas**
- ✅ Feature scaling / normalization
- ✅ Handling categorical variables
- ✅ Calculating and minimizing **MSE loss**
- ✅ Data visualization using Matplotlib

---

## 🔁 Features Used

Depending on your implementation, these might include:
- `Present_Price`
- `Kms_Driven`
- `Owner`
- `Year` → converted to **Car Age**
- Categorical encoding of:
  - `Fuel_Type`
  - `Seller_Type`
  - `Transmission`

---

## 🧠 Model Summary

```text
Gradient Descent Parameters:
  - Epochs: 2000
  - Learning Rate: 0.01
