# Multiple Disease Prediction System

## User Manual

This application is a health assistant that predicts the likelihood of various diseases using machine learning models. The diseases included are:

- Diabetes
- Heart Disease
- Parkinson's Disease
- Breast Cancer

### How to Use

1. Run the Streamlit app by executing:
   ```
   streamlit run app.py
   ```
2. Use the sidebar menu to select the disease prediction you want to perform.
3. Enter the required input parameters for the selected disease in the provided fields.
4. Click the "Test Result" button to get the prediction.
5. The result will be displayed below the button indicating whether the person is likely to have the disease or not.

Please ensure to enter valid numeric values for all input fields. For categorical fields in Kidney Disease prediction, select the appropriate option from the dropdown.

## Technologies Used

- Python 3.10
- Streamlit for the web interface
- Scikit-learn for machine learning models
- Pickle for model serialization
- Base64 for encoding images in the UI

## Algorithms Used

- Logistic Regression (Diabetes, Heart Disease)
- Support Vector Machine (Parkinson's Disease)
- Random Forest / Other classifiers (Kidney Disease, Breast Cancer) *(based on the saved models)*

## Accuracy Rate

- Diabetes Prediction: Approximately 78-80% accuracy
- Heart Disease Prediction: Approximately 85% accuracy
- Parkinson's Disease Prediction: Approximately 85-90% accuracy
- Breast Cancer Prediction: Approximately 92-95% accuracy

*Note: Accuracy rates are based on the training datasets used for the models and may vary in real-world scenarios.*

## How to Run the Application

1. Ensure Python 3.10 or higher is installed.
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Place the saved model files in the `saved_models` directory:
   - diabetes_model.sav
   - heart_disease_model.sav
   - parkinsons_model.sav
   - kidney.pkl
   - breast_cancer_model.sav
4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
5. Access the app in your browser at the URL provided in the terminal (usually http://localhost:8501).

## Additional Notes

- The app includes input validation to ensure correct data entry.
- Background images and custom icons are used for better UI experience.
- The models were trained on publicly available datasets and serialized using pickle.

## Modules Overview

This project consists of multiple disease prediction modules integrated into a single Streamlit application:

- **Diabetes Prediction:** Uses a machine learning model to predict the likelihood of diabetes based on user input parameters such as glucose level, blood pressure, BMI, and age. The model is loaded from a saved file and used for real-time prediction.

- **Heart Disease Prediction:** Predicts the presence of heart disease using clinical parameters like age, chest pain type, cholesterol level, and exercise-induced angina. The model is pre-trained and loaded for inference.

- **Parkinson's Disease Prediction:** Utilizes voice measurement features to predict Parkinson's disease. The model is trained on relevant datasets and integrated into the app for user input-based prediction. Input data is scaled before prediction using a saved scaler.

- **Breast Cancer Prediction:** Uses features extracted from breast cancer datasets to classify tumors as malignant or benign. The model is loaded and used to provide predictions based on user inputs.

All these modules rely on saved machine learning models located in the `saved_models` directory and are accessible through the app's sidebar menu for easy navigation and use.
