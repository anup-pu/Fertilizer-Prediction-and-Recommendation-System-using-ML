import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your data
fertilizer_data = pd.read_csv('Fertilizer.csv')

# Create label encoders
le_soil = LabelEncoder()
le_crop = LabelEncoder()

# Fit label encoders
le_soil.fit(fertilizer_data['Soil Type'])
le_crop.fit(fertilizer_data['Crop Type'])

# Save label encoders
joblib.dump(le_soil, 'soil_label_encoder.pkl')
joblib.dump(le_crop, 'crop_label_encoder.pkl')

print("Label encoders saved successfully.")
