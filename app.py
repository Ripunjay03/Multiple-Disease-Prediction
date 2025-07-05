import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import base64

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Get working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load saved models
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
diabetes_scaler = pickle.load(open(f'{working_dir}/saved_models/diabetes_scaler.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
heart_disease_scaler = pickle.load(open(f'{working_dir}/saved_models/heart_disease_scaler.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))
parkinsons_scaler = pickle.load(open(f'{working_dir}/saved_models/parkinsons_scaler.sav', 'rb'))
breastcancer_model = pickle.load(open(f'{working_dir}/saved_models/breast_cancer_model.sav', 'rb'))

# Sidebar navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Home', 'Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Breast Cancer Prediction'],
                           menu_icon='hospital-fill',
                           icons=['house', 'activity', 'heart', 'person', 'gender-female'],
                           default_index=0,
                            styles={
            "container": {"padding": "5px", "background-color": "#867A7A"},
            "icon": {"color": "red", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "color": "white",
                "padding": "10px 15px"
            },
            "nav-link-selected": {
                "background-color": "white",
                "color": "black",
                "font-weight": "bold"
            }
        }
    )
    

import base64

# Home Page
if selected == 'Home':
    # Load background image and convert to base64
    image_path = "C:/Users/Ripunjay Raj/Desktop/Streamlit/background image.jpg"
    with open(image_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode()

    # Inject CSS for background
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
    }}
    .stMarkdown {{
        color: white;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Content
    st.title("Welcome to the Health Assistant!!!")
    st.write("""
        This application is designed to assist you in predicting various health conditions using machine learning models.

        You can navigate through the options in the sidebar to predict:
        - Diabetes
        - Heart Disease
        - Parkinson's Disease
        - Breast Cancer

        Each section will guide you through the necessary inputs required for the prediction.

        Please ensure that you provide accurate information for the best results.
    """)


# ======================== DIABETES PAGE ========================
import pandas as pd

if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    col1, col2, col3 = st.columns(3)
    with col1: Pregnancies = st.text_input('Number of Pregnancies')
    with col2: Glucose = st.text_input('Glucose Level')
    with col3: BloodPressure = st.text_input('Blood Pressure')
    with col1: SkinThickness = st.text_input('Skin Thickness')
    with col2: Insulin = st.text_input('Insulin Level')
    with col3: BMI = st.text_input('BMI')
    with col1: DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    with col2: Age = st.text_input('Age')

    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        try:
            input_data = []
            for val in [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]:
                if val.strip() == '':
                    st.error("All input fields are required and must be numeric.")
                    break
                try:
                    input_data.append(float(val))
                except ValueError:
                    st.error(f"Invalid input for value: {val}. Please enter numeric values only.")
                    break
            else:
                # Convert input_data list to DataFrame with column names
                columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
                input_df = pd.DataFrame([input_data], columns=columns)
                input_data_scaled = diabetes_scaler.transform(input_df)
                prediction = diabetes_model.predict(input_data_scaled)
                diab_diagnosis = 'The person is diabetic' if prediction[0] == 1 else 'The person is not diabetic'
        except ValueError:
            st.error("Please enter valid numeric values.")

    st.success(diab_diagnosis)

# ======================== HEART DISEASE PAGE ========================
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    col1, col2, col3 = st.columns(3)
    with col1: age = st.text_input('Age')
    with col2: sex = st.text_input('Sex (1=Male, 0=Female)')
    with col3: cp = st.text_input('Chest Pain Type')
    with col1: trestbps = st.text_input('Resting Blood Pressure')
    with col2: chol = st.text_input('Cholesterol')
    with col3: fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1 = True; 0 = False)')
    with col1: restecg = st.text_input('Resting ECG Results')
    with col2: thalach = st.text_input('Max Heart Rate Achieved')
    with col3: exang = st.text_input('Exercise Induced Angina')
    with col1: oldpeak = st.text_input('ST Depression')
    with col2: slope = st.text_input('Slope of Peak Exercise ST Segment')
    with col3: ca = st.text_input('Number of Major Vessels Colored')
    with col1: thal = st.text_input('Thal (0=Normal, 1=Fixed Defect, 2=Reversible)')

    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        try:
            input_data_raw = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
                             oldpeak, slope, ca, thal]
            for val in input_data_raw:
                if val.strip() == '':
                    st.error("All input fields are required and must be numeric.")
                    break
            else:
                input_data = []
                for val in input_data_raw:
                    try:
                        input_data.append(float(val))
                    except ValueError:
                        st.error(f"Invalid input for value: {val}. Please enter numeric values only.")
                        break
                else:
                    import pandas as pd
                    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
                               'oldpeak', 'slope', 'ca', 'thal']
                    input_df = pd.DataFrame([input_data], columns=columns)
                    input_data_scaled = heart_disease_scaler.transform(input_df)
                    prediction = heart_disease_model.predict(input_data_scaled)
                    heart_diagnosis = 'The person has heart disease' if prediction[0] == 1 else 'The person does not have heart disease'
        except:
            st.error("Please enter valid numeric values.")
    st.success(heart_diagnosis)

# ======================== PARKINSON'S PAGE ========================
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")
    cols = st.columns(5)
    field_names = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
                   'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
                   'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
                   'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
    inputs = []
    for i, name in enumerate(field_names):
        with cols[i % 5]:
            inputs.append(st.text_input(name))

    parkinsons_diagnosis = ''
    if st.button("Parkinson's Test Result"):
        try:
            for val in inputs:
                if val.strip() == '':
                    st.error("All input fields are required and must be numeric.")
                    break
            else:
                input_data = []
                for val in inputs:
                    try:
                        input_data.append(float(val))
                    except ValueError:
                        st.error(f"Invalid input for value: {val}. Please enter numeric values only.")
                        break
                else:
                    import pandas as pd
                    columns = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
                               'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
                               'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
                               'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
                    input_df = pd.DataFrame([input_data], columns=columns)
                    input_data_scaled = parkinsons_scaler.transform(input_df)
                    prediction = parkinsons_model.predict(input_data_scaled)
                    parkinsons_diagnosis = "The person has Parkinson's disease" if prediction[0] == 1 else "The person does not have Parkinson's disease"
        except:
            st.error("Please enter valid numeric values.")
    st.success(parkinsons_diagnosis)

# ======================== BREAST CANCER PAGE ========================
if selected == "Breast Cancer Prediction":
    st.title('Breast Cancer Prediction using ML')
    
    cols = st.columns(3)
    field_names = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
        'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se',
        'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
        'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
        'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
        'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    inputs = []
    for i, name in enumerate(field_names):
        with cols[i % 3]:
            inputs.append(st.text_input(name.replace("_", " ").title()))

    cancer_result = ''
    if st.button('Breast Cancer Test Result'):
        try:
            for val in inputs:
                if val.strip() == '':
                    st.error("All input fields are required and must be numeric.")
                    break
            else:
                input_data = []
                for val in inputs:
                    try:
                        input_data.append(float(val))
                    except ValueError:
                        st.error(f"Invalid input for value: {val}. Please enter numeric values only.")
                        break
                else:
                    import pandas as pd
                    columns = [
                        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
                        'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se',
                        'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
                        'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
                        'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
                        'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
                    ]
                    input_df = pd.DataFrame([input_data], columns=columns)
                    prediction = breastcancer_model.predict(input_df)
                    cancer_result = 'The Breast Cancer is Malignant' if prediction[0] == 0 else 'The Breast Cancer is Benign'
        except:
            st.error("Please enter valid numeric values.")
    st.success(cancer_result)
