from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import joblib
import requests

app = Flask(__name__)

# Load models and encoders
clf = joblib.load('fertilizer_classifier_model.pkl')
scaler_class = joblib.load('fertilizer_classifier_scaler.pkl')
le_fertilizer = joblib.load('fertilizer_label_encoder.pkl')

reg = joblib.load('nutrient_regressor_model.pkl')
scaler_reg = joblib.load('nutrient_regressor_scaler.pkl')

# Load encoders for soil and crop types
le_soil = joblib.load('soil_label_encoder.pkl')
le_crop = joblib.load('crop_label_encoder.pkl')

# Replace with your active WeatherAPI key
BASE_URL = 'http://api.weatherapi.com/v1'
API_KEY = 'f8c0f073016348d885582046240109'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_weather', methods=['POST'])
def get_weather():
    try:
        data = request.json
        latitude = data.get('latitude')
        longitude = data.get('longitude')

        if latitude is not None and longitude is not None:
            # Construct the API URL with latitude and longitude
            url = f'{BASE_URL}/current.json?key={API_KEY}&q={latitude},{longitude}'
        else:
            location = data.get('location', 'Delhi, India')  # Fallback to default if no coordinates
            url = f'{BASE_URL}/current.json?key={API_KEY}&q={location}'

        response = requests.get(url)
        response.raise_for_status()

        weather_data = response.json()

        # Print the full JSON response for debugging
        print(weather_data)  # Debugging line

        # Extract weather information
        location_name = weather_data['location']['name']
        country = weather_data['location']['country']
        temp_c = weather_data['current']['temp_c']
        condition_text = weather_data['current']['condition']['text']
        humidity = weather_data['current']['humidity']
        wind_speed_kph = weather_data['current']['wind_kph']

        return jsonify({
            'location': f'{location_name}, {country}',
            'temp_c': temp_c,
            'condition': condition_text,
            'humidity': humidity,
            'wind_speed_kph': wind_speed_kph
        })
        
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f"Error fetching weather data: {e}"}), 500
    except ValueError as e:
        return jsonify({'error': f"Error parsing JSON response: {e}"}), 500
    except KeyError as e:
        return jsonify({'error': f"Error fetching weather data: Missing key {e}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        # Get weather data
        temperature = float(data.get('temperature', 0))
        humidity = float(data.get('humidity', 0))

        # If temperature or humidity is not provided by the user, fetch from the weather API
        if temperature == 0 or humidity == 0:
            location = data.get('location', 'Delhi, India')  # Default to 'Delhi, India' if no location provided
            
            # Construct the API URL for WeatherAPI
            url = f'{BASE_URL}/current.json?key={API_KEY}&q={location}'
            
            response = requests.get(url)
            response.raise_for_status()  # Check if the request was successful

            # Parse the JSON response
            weather_data = response.json()

            # Extract weather information
            temperature = weather_data['current']['temp_c']
            humidity = weather_data['current']['humidity']

        ph = float(data.get('ph', 0))
        rainfall = float(data.get('rainfall', 0))
        moisture = float(data.get('moisture', 0))
        soil_type = data.get('soil_type', 'Loamy')  # Default value 'Loamy' if not provided
        crop_type = data.get('crop_type', 'Wheat')  # Default value 'Wheat' if not provided
        nitrogen = float(data.get('nitrogen', 0))
        potassium = float(data.get('potassium', 0))
        phosphorous = float(data.get('phosphorous', 0))

        # Prepare data for classification
        input_class = pd.DataFrame([[temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous]],
                                   columns=['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous'])
        
        input_class['Soil Type'] = le_soil.transform([soil_type])
        input_class['Crop Type'] = le_crop.transform([crop_type])

        # Scale input features for classifier
        input_class_scaled = scaler_class.transform(input_class)

        # Predict fertilizer type
        fertilizer_prediction = clf.predict(input_class_scaled)
        fertilizer_prediction = le_fertilizer.inverse_transform(fertilizer_prediction)[0]

        # Prepare data for regression
        input_reg = pd.DataFrame([[temperature, humidity, ph, rainfall, nitrogen, potassium, phosphorous]],
                                 columns=['temperature', 'humidity', 'ph', 'rainfall', 'N', 'P', 'K'])

        # Scale input features for regression
        input_reg_scaled = scaler_reg.transform(input_reg)

        # Predict nutrient requirements
        nutrient_prediction = reg.predict(input_reg_scaled)

        # Ensure nutrient_prediction is an array with exactly 3 elements
        if isinstance(nutrient_prediction, np.ndarray):
            nutrient_prediction = nutrient_prediction.flatten()  # Ensure it's a 1D array
            if nutrient_prediction.shape[0] < 3:
                nutrient_prediction = np.pad(nutrient_prediction, (0, 3-len(nutrient_prediction)), mode='constant')
        else:
            nutrient_prediction = np.array([0, 0, 0])

        # Define the % nutrient content in the fertilizers
        fertilizer_content = {
            'Urea': {'N': 46.00, 'P': 0.00, 'K': 0.00},
            'DAP': {'N': 18.00, 'P':46.00, 'K': 0.00},
            '14-35-14': {'N': 14.00, 'P': 35.00, 'K': 14.00},
            '28-28': {'N': 28.00, 'P': 28.00, 'K': 0.00},
            '17-17-17': {'N': 17.00, 'P': 17.00, 'K': 17.00},
            '20-20': {'N': 20.00, 'P': 20.00, 'K': 0.00},
            '10-26-26': {'N': 10.00, 'P': 26.00, 'K': 26.00}
        }

        # Calculate the amount of fertilizer needed
        if fertilizer_prediction in fertilizer_content:
            content = fertilizer_content[fertilizer_prediction]
            amount_needed = {
                'N': (nutrient_prediction[0] / content['N']) * 100 if content['N'] > 0 else 0,
                'P': (nutrient_prediction[1] / content['P']) * 100 if content['P'] > 0 else 0,
                'K': (nutrient_prediction[2] / content['K']) * 100 if content['K'] > 0 else 0,
            }

            # Calculate the total fertilizer amount needed
            total_fertilizer_amount = max(amount_needed.values(), default=0)
        else:
            total_fertilizer_amount = "Fertilizer not in predefined list"

        return jsonify({
    'temperature': temperature,
    'humidity': humidity,
    'ph': ph,
    'rainfall': rainfall,
    'moisture': moisture,
    'soil_type': soil_type,
    'crop_type': crop_type,
    'nitrogen': nitrogen,
    'phosphorous': phosphorous,
    'potassium': potassium,
    'fertilizer': fertilizer_prediction,
    'nutrient': nutrient_prediction.tolist(),
    'amount_needed': total_fertilizer_amount
})

    
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
