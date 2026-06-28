# Car Price Prediction using Machine Learning 🚗💰

## Project Overview

This project predicts the selling price of a used car using Machine Learning. The model analyzes various car features such as manufacturing year, present price, kilometers driven, fuel type, transmission type, selling type, and ownership history to estimate the resale price.

## Objective

The main objective of this project is to build a machine learning model that can accurately predict the selling price of a car based on its features.

## Dataset Features

The dataset contains the following columns:

* **Car_Name** → Name of the car
* **Year** → Manufacturing year
* **Selling_Price** → Selling price of car (Target Variable)
* **Present_Price** → Current showroom price
* **Driven_kms** → Total kilometers driven
* **Fuel_Type** → Petrol / Diesel / CNG
* **Selling_type** → Dealer / Individual
* **Transmission** → Manual / Automatic
* **Owner** → Number of previous owners

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Linear Regression
* VS Code

## Machine Learning Algorithm

This project uses **Linear Regression** to predict car prices.

## Project Workflow

1. Load dataset
2. Analyze data
3. Handle categorical values using encoding
4. Select features and target variable
5. Split data into training and testing sets
6. Train Linear Regression model
7. Evaluate model accuracy
8. Predict car selling price using user input

## Model Accuracy

* **R² Score:** 0.85

This means the model predicts car prices with approximately **85% accuracy**.

## Sample Prediction

User Input:

* Year: 2018
* Present Price: 8.5 Lakhs
* Driven Kms: 30000
* Fuel Type: Petrol
* Selling Type: Individual
* Transmission: Manual
* Owner: 0

Predicted Output:

* **Selling Price: ~5-6 Lakhs**

## Key Features

✅ Data preprocessing
✅ Feature encoding
✅ Model training
✅ Accuracy evaluation
✅ Interactive user input prediction

## Future Improvements

* Build GUI/Desktop application
* Create web app using Streamlit or Flask
* Use advanced ML models for better accuracy

## Author

**Gourav Singh**
B.Tech CSE Student | Vivekanand Global University (VGU) by Sunstone
Data Science & Machine Learning Intern at Oasis Infobyte
