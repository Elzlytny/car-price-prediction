# Car Price Prediction Using Machine Learning

## Project Overview
This project focuses on predicting used car prices using multiple machine learning regression algorithms. The objective is to analyze vehicle-related features, compare different regression techniques, and identify the most effective model for accurate price prediction.

The best-performing model, Random Forest Regressor, was selected and deployed through a desktop graphical user interface (GUI) built with Tkinter.

---

## App Screenshots
![Main Interface](screenshots/app_Screenshot1.png)
![Prediction Result](screenshots/app_Screenshot2.png)

---

## Problem Statement
Car pricing depends on multiple technical and market-related factors such as manufacturing year, mileage, engine capacity, and drivetrain type. The goal of this project is to build predictive models capable of estimating vehicle prices based on historical data.

---

## Dataset Features
The dataset includes the following attributes:

- Year
- Kilometer
- Engine
- Max Power
- Max Torque
- Owner
- Length
- Width
- Height
- Seating Capacity
- Fuel Tank Capacity
- Drivetrain

Target Variable:
- Price

---

## Data Preprocessing
The following preprocessing steps were applied:

- Cleaning and converting engine, power, and torque values to numerical format
- Handling missing values using median and mode imputation
- Encoding categorical variables using one-hot encoding
- Splitting data into training (80%) and testing (20%) sets
- Feature scaling for distance-based algorithms (KNN, SVR)

---

## Models Implemented
The following regression algorithms were trained and evaluated:

- Linear Regression
- K-Nearest Neighbors Regressor
- Random Forest Regressor
- Support Vector Regressor
- Decision Tree Regressor

---

## Model Evaluation
Models were evaluated using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Best Performing Model
Random Forest Regressor achieved:

- R² Score (Test): 0.826
- RMSE: 1,100,096

The Decision Tree model showed signs of overfitting, while Random Forest demonstrated better generalization performance on unseen data.

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Tkinter
- Joblib

---

## GUI Application
A desktop graphical interface was developed using Tkinter to allow users to:

- Enter vehicle specifications
- Predict car prices instantly
- Display formatted price results

The trained model and feature columns were saved using Joblib for deployment.

---

## How to Run the Project

1. Install required dependencies:
pip install -r requirements.txt

2. Run the GUI application:
python gui.py

---

## Author
Mohammed Elzlytny  
Junior Penetration Tester | Computers & AI Student
