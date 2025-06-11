# REAL-ESTATE-PRICE-PREDICTION
Overview
This project aims to predict the price per unit area of real estate properties using supervised machine learning. By leveraging key features such as house age, proximity to public transport, and number of nearby amenities, the model enables data-driven property valuation. This can empower buyers, sellers, and agents to make more informed decisions in real estate markets.

Problem Statement
In real estate, price estimation often relies on subjective judgment or outdated comparative listings. This leads to inconsistencies and inefficient negotiations. Our goal is to build a machine learning model that automates the prediction of property prices with improved accuracy, consistency, and scalability. By training on historical data, the model can uncover patterns that correlate strongly with property values.

Dataset
The dataset (Real estate.csv) contains structured information on property listings:

Feature	Description
No	Serial number (not used in modeling)
X1 transaction date	Date of transaction (year/month)
X2 house age	Age of the house in years
X3 distance to MRT station	Distance to nearest metro station (meters)
X4 number of convenience stores	Count of nearby stores
X5 latitude	Geographical coordinate
X6 longitude	Geographical coordinate
Y house price of unit area	💡 Target: House price per unit area

Technologies Used
Python 3.x

Pandas, NumPy – Data manipulation

Matplotlib, Seaborn – Data visualization

Scikit-learn – Machine learning (Linear Regression)

Jupyter Notebook – Interactive development environment

Project Workflow
1. Data Loading & Exploration
Imported dataset and examined structure

Visualized feature relationships with pair plots and distribution curves

2. Data Preprocessing
Dropped irrelevant features (e.g., No)

Defined input features X and target variable y

3. Model Training
Split dataset into training (70%) and testing (30%) sets

Trained a Linear Regression model using Scikit-learn

4. Model Evaluation
Extracted regression coefficients and intercept

(Future enhancement: Evaluate with R², MAE, RMSE)

Results
Successfully trained a regression model on real estate data

Produced interpretable coefficients to understand feature importance

Established a baseline model for price prediction

Future Improvements
Add model performance metrics: R² score, MAE, RMSE
Compare with other regression models: Ridge, Lasso, XGBoost
Build a Streamlit or Flask web app for live predictions
Deploy the model using Heroku, Render, or AWS Lambda
