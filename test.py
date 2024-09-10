import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Load the data
data = pd.read_csv("readmission_data.csv")

# Define the feature columns
features = [
    "Age",
    "Gender",
    "Diagnosis",
    "Length_of_Stay",
    "Prescribed_Medication"
]

# Select relevant columns, drop NA values
model_data = data[features + ["Readmitted"]]
model_data.dropna(inplace=True)

# Initialize LabelEncoders for categorical features
label_encoders = {}
for col in features:
    if model_data[col].dtype == "object":
        label_encoders[col] = LabelEncoder()
        model_data[col] = label_encoders[col].fit_transform(model_data[col])

# Separate features (X) and target (y)
X = model_data[features]
y = model_data["Readmitted"]

# Split the data before scaling to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the continuous features on training data only
scaler = StandardScaler()
X_train[["Age", "Length_of_Stay"]] = scaler.fit_transform(X_train[["Age", "Length_of_Stay"]])
X_test[["Age", "Length_of_Stay"]] = scaler.transform(X_test[["Age", "Length_of_Stay"]])

# Handle class imbalance using SMOTE on the training data only
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train a RandomForestClassifier with balanced class weights
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf_model.fit(X_train_res, y_train_res)


# Predict on training and test sets
y_train_pred = rf_model.predict(X_train_res)
y_test_pred = rf_model.predict(X_test)

# Calculate accuracy for training and test sets
train_accuracy = accuracy_score(y_train_res, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Display the results
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Make predictions and print classification report
print(classification_report(y_test, y_test_pred))

# Function to encode new input data
def encoding_input_data(age, gender, diagnosis, length_of_stay, prescribed_medication):
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Diagnosis": diagnosis,
        "Length_of_Stay": length_of_stay,
        "Prescribed_Medication": prescribed_medication
    }])
    
    # Encode the input data using the label encoders
    for column in features:
        if column in label_encoders:
            # Ensure the value exists in the encoder to avoid unseen value errors
            if input_data[column].iloc[0] not in label_encoders[column].classes_:
                raise ValueError(f"Value '{input_data[column].iloc[0]}' for '{column}' not seen in training data.")
            input_data[column] = label_encoders[column].transform(input_data[column])
    
    # Standardize the numerical columns
    input_data[["Age", "Length_of_Stay"]] = scaler.transform(input_data[["Age", "Length_of_Stay"]])
    
    return input_data

# Example prediction
try:
    prediction = rf_model.predict(encoding_input_data(50, "Female", "COPD", 88, "Lisinopril"))
    print(f"Prediction - {prediction}")
except ValueError as e:
    print(e)
