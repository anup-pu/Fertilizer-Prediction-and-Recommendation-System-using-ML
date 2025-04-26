# Fertilizer-Prediction-and-Recommendation-System-using-ML
This project predicts the type of fertilizer and amount needed for crops based on soil, crop, and weather conditions. It also integrates real-time weather data using the WeatherAPI.




## Features
- Predict suitable fertilizer type.
- Predict fertilizer amount required (N, P, K nutrients).
- Real-time weather data fetching based on location or coordinates.
- Built with Flask, scikit-learn, and WeatherAPI.

## Tech Stack
- Python, Flask
- Scikit-learn (ML models)
- HTML (for frontend)
- WeatherAPI (for weather data)

## How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your WeatherAPI key in the code.
4. Run the app:
   ```bash
   python app.py
   ```
5. Visit `http://localhost:5000` in your browser.

## Endpoints
- `/` - Home page
- `/get_weather` - Get weather details via API
- `/predict` - Predict fertilizer type and amount

## Note
- Make sure to include all `.pkl` model and encoder files in the project directory.

---

![Screenshot (26)](https://github.com/user-attachments/assets/c243d39f-35df-4d73-a852-f3011c2037cc)
![Screenshot (25)](https://github.com/user-attachments/assets/014a9cf5-2875-4b57-be70-fe0a83c5a06e)
