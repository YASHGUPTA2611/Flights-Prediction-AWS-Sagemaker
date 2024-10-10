# Flight Price Prediction: End-to-End Machine Learning Project Using AWS SageMaker

## 🌐 Live Streamlit App
[Access the Flight Price Prediction App](https://flights-price-prediction-app-deployment-orwdbbtjcrxfoxrfbgv8dx.streamlit.app/)

<img src="additional_files/first-ezgif.com-video-to-gif-converter.gif" width="800" height="500" />

## 📊 Overview
This repository hosts a Streamlit web application designed to predict flight ticket prices using machine learning models. Users can input various flight details—including airline, journey date, source, destination, and departure time—and receive a predicted ticket price. The application employs a pre-trained model with comprehensive preprocessing pipelines using Scikit-learn, Feature-engine, and a RandomForestRegressor for feature selection, with an XGBoost model for final price predictions.

## ✨ Features
- **User Inputs:** Easily select flight details such as airline, source, destination, departure and arrival times, duration, and total stops.
- **Preprocessing Pipelines:** Data transformations are seamlessly handled using Scikit-learn pipelines, including:
  - Rare label encoding and mean encoding for categorical variables
  - Custom feature engineering (e.g., source-destination mapping and time of day extraction)
  - Scaling features using techniques like StandardScaler and PowerTransformer
- **Model:** A pre-trained XGBoost model provides accurate flight price predictions based on user inputs.
- **Web Interface:** A user-friendly and interactive interface created with Streamlit.

## 🛠️ Dependencies
Before running the application, ensure you have packages installed from requirements.txt

## Code Walkthrough
### Preprocessing Pipelines
The code employs ColumnTransformer and Pipeline to effectively handle preprocessing, including:

Encoding airlines with RareLabelEncoder and MeanEncoder
Extracting journey date features using DatetimeFeatures
Combining and encoding source and destination with custom transformers and distance mapping
Processing departure and arrival times to determine the time of day
Scaling features using StandardScaler and PowerTransformer
### Feature Selection
A RandomForestRegressor serves as the base estimator in the SelectBySingleFeaturePerformance selector to identify relevant features based on the R² score.

### Model Integration
The preprocessor is fitted and saved as preprocessor.joblib, while the XGBoost model is loaded during prediction using pickle.

### Web Application
User inputs are gathered via Streamlit widgets and transformed using the saved preprocessor. The predicted flight price is displayed on the webpage using the pre-trained XGBoost model.
