import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

# Load datasets
crop_data = pd.read_csv('Crop_dataset.csv')
fertilizer_data = pd.read_csv('Fertilizer.csv')

# Strip extra spaces from column names
fertilizer_data.columns = fertilizer_data.columns.str.strip()

# Print column names for debugging
print("Fertilizer Data Columns:", fertilizer_data.columns.tolist())
print("Crop Data Columns:", crop_data.columns.tolist())

# Handle missing values using ffill and bfill methods
crop_data.ffill(inplace=True)
fertilizer_data.ffill(inplace=True)

# Encode categorical variables
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fertilizer = LabelEncoder()

fertilizer_data['Soil Type'] = le_soil.fit_transform(fertilizer_data['Soil Type'])
fertilizer_data['Crop Type'] = le_crop.fit_transform(fertilizer_data['Crop Type'])
fertilizer_data['Fertilizer Name'] = le_fertilizer.fit_transform(fertilizer_data['Fertilizer Name'])

# Features and target variable for classification (fertilizer type prediction)
X_class = fertilizer_data[['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
y_class = fertilizer_data['Fertilizer Name']

# Splitting data into training and testing sets for classification
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Standardize the features
scaler_class = StandardScaler()
X_train_class_scaled = scaler_class.fit_transform(X_train_class)
X_test_class_scaled = scaler_class.transform(X_test_class)

# Train the classification model (Random Forest Classifier)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_class_scaled, y_train_class)

# Save the classification model and scaler
joblib.dump(clf, 'fertilizer_classifier_model.pkl')
joblib.dump(scaler_class, 'fertilizer_classifier_scaler.pkl')
joblib.dump(le_fertilizer, 'fertilizer_label_encoder.pkl')

# Check and convert non-numeric values in the target variable for regression
print("Unique values in 'label' before conversion:", crop_data['label'].unique())
if crop_data['label'].dtype == 'object':
    le_label = LabelEncoder()
    crop_data['label'] = le_label.fit_transform(crop_data['label'])
    print("Unique values in 'label' after conversion:", crop_data['label'].unique())

# Features and target variable for regression (nutrient requirement estimation)
X_reg = crop_data[['temperature', 'humidity', 'ph', 'rainfall', 'N', 'P', 'K']]
# Ensure target variable y_reg contains separate columns for N, P, and K
y_reg = crop_data[['N', 'P', 'K']]

# Splitting data into training and testing sets for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Standardize the features for regression
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# Train the regression model (Random Forest Regressor)
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_reg_scaled, y_train_reg)

# Save the regression model and scaler
joblib.dump(reg, 'nutrient_regressor_model.pkl')
joblib.dump(scaler_reg, 'nutrient_regressor_scaler.pkl')

print("Models trained and saved successfully.")
