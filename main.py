import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Set up Streamlit page configuration
st.set_page_config(page_title="Healthcare Data Analysis and Prediction System", page_icon="📊", layout="wide")

st.markdown("""
<style>
    header {
        display:none!important-;
    }
</style>
""", unsafe_allow_html=True)

# Define the user credentials
USERNAME = "venus"
PASSWORD = "venus123"

# Initialize session state for authentication
if 'authentication_status' not in st.session_state:
    st.session_state.authentication_status = None

# Function to authenticate the user
def authenticate(username, password):
    if username == USERNAME and password == PASSWORD:
        st.session_state.authentication_status = True
        st.rerun()
    else:
        st.session_state.authentication_status = False
        st.error("Invalid username or password")

# Function to display the logout button after login
def login_logout():
    if st.session_state.authentication_status:
        st.sidebar.button("Logout", on_click=logout)
    else:
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            authenticate(username, password)

# Function to handle logout
def logout():
    st.session_state.authentication_status = False

# Main page layout
def main_page():
    st.title("Healthcare Data Analysis and Prediction System")

    # Display login or logout based on authentication status
    login_logout()

    if st.session_state.authentication_status:
        visualize_and_predict_page()
    else:
        st.image("assets//img-3-7-1024x614.png")
        st.html("<h5 style='text-align:center;color:red'>Please sign in to use the system</h5>")
        st.html("<span style='text-align:center;color:#414eb4;display:block'>Developed by Venus Gonyora @ 2024</span>")

# Visualization and Prediction page
def visualize_and_predict_page():
    st.sidebar.title("Upload and Filter Data")

    # Sidebar for file upload
    uploaded_file = st.sidebar.file_uploader("Upload EHR CSV Data", type="csv")

    # Check if a file is uploaded
    if uploaded_file:
        # Load data using pandas
        data = pd.read_csv(uploaded_file)

        # Add Year column from Date_of_Visit
        data['Date_of_Visit'] = pd.to_datetime(data['Date_of_Visit'])
        data['Year'] = data['Date_of_Visit'].dt.year

        # Sidebar filters
        st.sidebar.subheader('Filter the Data')
        year_options = data['Year'].unique()
        selected_years = st.sidebar.multiselect("Select Years", year_options, default=year_options)

        medication_options = data['Prescribed_Medication'].unique()
        selected_medications = st.sidebar.multiselect("Select Medications", medication_options, default=medication_options)

        diagnosis_options = data['Diagnosis'].unique()
        selected_diagnoses = st.sidebar.multiselect("Select Diagnoses", diagnosis_options, default=diagnosis_options)

        gender_options = data['Gender'].unique()
        selected_genders = st.sidebar.multiselect("Select Genders", gender_options, default=gender_options)

        region_options = data['Region'].unique()
        selected_regions = st.sidebar.multiselect("Select Regions", region_options, default=region_options)

        # Filter data based on selections
        filtered_data = data[
            (data['Year'].isin(selected_years)) &
            (data['Prescribed_Medication'].isin(selected_medications)) &
            (data['Diagnosis'].isin(selected_diagnoses)) &
            (data['Gender'].isin(selected_genders)) &
            (data['Region'].isin(selected_regions))
        ]

        # Dashboard Layout
        st.write("### Key Performance Indicators (KPIs)")

        total_patients = filtered_data['Patient_ID'].nunique()
        total_visits = filtered_data.shape[0]
        avg_age = filtered_data['Age'].mean()
        most_common_diagnosis = filtered_data['Diagnosis'].mode()[0]
        most_prescribed_medication = filtered_data['Prescribed_Medication'].mode()[0]

        # Display KPIs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Patients", total_patients)
        col2.metric("Total Visits", total_visits)
        col3.metric("Average Age", round(avg_age, 2))
        col4.metric("Most Common Diagnosis", most_common_diagnosis)
        col4.metric("Most Prescribed Medication", most_prescribed_medication)

        # Create sections for visualizations
        col1, col2 = st.columns(2)

        with col1:
            # Demographics Dashboard
            st.subheader('Demographics')
            age_fig = px.histogram(filtered_data, x='Age', nbins=10, title='Age Distribution')
            st.plotly_chart(age_fig, use_container_width=True)

            gender_fig = px.pie(filtered_data, names='Gender', title='Gender Distribution')
            st.plotly_chart(gender_fig, use_container_width=True)

            # Diagnosis Analysis Dashboard
            st.subheader('Diagnosis Analysis')
            diagnosis_counts = filtered_data['Diagnosis'].value_counts().reset_index()
            diagnosis_counts.columns = ['Diagnosis', 'count']  # Rename columns for clarity

            diagnosis_fig = px.bar(diagnosis_counts, 
                                x='Diagnosis', y='count', 
                                title='Diagnosis Frequency', 
                                labels={'Diagnosis': 'Diagnosis', 'count': 'Frequency'})

            st.plotly_chart(diagnosis_fig, use_container_width=True)

            # Medication Prescribing Dashboard
            st.subheader('Medication Prescribing')
            medication_counts = filtered_data['Prescribed_Medication'].value_counts().reset_index()
            medication_counts.columns = ['Medication', 'count']  # Rename columns for clarity

            medication_fig = px.bar(medication_counts, x='Medication', y='count', title='Medication Frequency', labels={'Medication': 'Medication', 'count': 'Frequency'})
            st.plotly_chart(medication_fig, use_container_width=True)

        with col2:
             # Diagnosis by Gender
            st.subheader('Diagnosis by Gender')
            diagnosis_gender_fig = px.bar(filtered_data.groupby(['Gender', 'Diagnosis']).size().reset_index(name='Counts'),
                                          x='Diagnosis', y='Counts', color='Gender', title='Diagnosis Frequency by Gender',
                                          labels={'Counts': 'Number of Diagnoses'})
            st.plotly_chart(diagnosis_gender_fig, use_container_width=True)

            # Patient Visits by Age Group
            st.subheader('Patient Visits by Age Group')
            age_groups = pd.cut(filtered_data['Age'], bins=[0, 18, 30, 40, 50, 60, 100], 
                                labels=['0-18', '19-30', '31-40', '41-50', '51-60', '60+'])
            age_group_fig = px.histogram(filtered_data, x=age_groups, title='Patient Visits by Age Group')
            st.plotly_chart(age_group_fig, use_container_width=True)

            # Medication Trends Over Time
            st.subheader('Medication Trends Over Time')
            selected_medication = st.selectbox("Select Medication to Explore", medication_options, index=0)
            med_trend_fig = px.line(filtered_data[filtered_data['Prescribed_Medication'] == selected_medication].groupby('Date_of_Visit').size().reset_index(name='Counts'),
                                    x='Date_of_Visit', y='Counts', title=f'Trend of {selected_medication} Over Time', labels={'Counts': 'Number of Prescriptions'})
            st.plotly_chart(med_trend_fig, use_container_width=True)

            # Medication by Region
            st.subheader('Medication by Region')
            medication_region_fig = px.bar(filtered_data.groupby('Region')['Prescribed_Medication'].value_counts().unstack().fillna(0).reset_index(),
                                           x='Region', y=filtered_data['Prescribed_Medication'].value_counts().index,
                                           title='Medication Frequency by Region', labels={'Prescribed_Medication': 'Medication'})
            st.plotly_chart(medication_region_fig, use_container_width=True)

        # Map for geographic visualization
        st.subheader('Geographic Data Map')
        if 'Latitude' in data.columns and 'Longitude' in data.columns:
            # Create the map with every user plotted
            map_fig = px.scatter_mapbox(filtered_data, lat="Latitude", lon="Longitude", hover_name="Region",
                                        hover_data=["First_Name", "Last_Name"], title="Map of Visits",
                                        zoom=3, height=600, width=1200)  # Increased width for better visibility
            map_fig.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(map_fig, use_container_width=True)
        else:
            st.error("Latitude and Longitude columns are required to plot the map.")

        # Machine Learning Model: Predicting Risk of Readmission
        st.subheader('Predict Patient Risk of Readmission')
        age_options = sorted(data['Age'].unique())

        gender_options = data['Gender'].unique()

        diagnosis_options = data['Diagnosis'].unique()

        length_of_stay_options = sorted(data['Length_of_Stay'].unique())

        medication_options = data['Prescribed_Medication'].unique()
        # Ensure we're using the full data instead of filtered data for the model
        # Feature columns definition
        feature_columns = ['Age', 'Gender', 'Diagnosis', 'Length_of_Stay', 'Prescribed_Medication']

        if 'Readmitted' in data.columns:
            # Use the original data frame for model data
            model_data = data[feature_columns + ['Readmitted']]
            model_data.dropna(inplace=True)

            # Encode categorical variables
            label_encoders = {}
            for column in feature_columns:
                if model_data[column].dtype == 'object':
                    label_encoders[column] = LabelEncoder()
                    model_data[column] = label_encoders[column].fit_transform(model_data[column])

            # Define features and target variable
            X = model_data[feature_columns]
            y = model_data['Readmitted']

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Train a RandomForestClassifier
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)

            # New Patient Prediction
            new_patient = {
                'Age': st.selectbox("Select Age", options=sorted(data['Age'].unique())),
                'Gender': st.selectbox("Select Gender", options=data['Gender'].unique()),
                'Diagnosis': st.selectbox("Select Diagnosis", options=data['Diagnosis'].unique()),
                'Length_of_Stay': st.selectbox("Select Length of Stay", options=sorted(data['Length_of_Stay'].unique())),
                'Prescribed_Medication': st.selectbox("Select Prescribed Medication", options=data['Prescribed_Medication'].unique())
            }
            
            if st.button("Predict Readmission Risk"):
                # Prepare input data for prediction
                input_data = pd.DataFrame([new_patient])[feature_columns]  # Ensure correct ordering
                
                # Encode the input data
                for column in feature_columns:
                    if column in label_encoders:
                        input_data[column] = label_encoders[column].transform(input_data[column])
                
                # Predict and display the result
                prediction = rf_model.predict(input_data)[0]
                if prediction == 1:
                    st.error("The patient is at **high risk** of readmission.")
                else:
                    st.success("The patient is at **low risk** of readmission.")

        else:
            st.warning("The dataset does not contain a 'Readmitted' column for predictive modeling.")

    else:
        st.info("Please upload a CSV file to visualize the data.")

# Run the main page
main_page()
