import pandas as pd
import joblib

model = joblib.load("crop_model.pkl")

sample = pd.DataFrame([
    {
        "district": "Anuradhapura",
        "crop": "Rice",
        "season": "Yala",
        "ph": 6.6,
        "nitrogen": 47,
        "phosphorus": 21,
        "potassium": 31,
        "rainfall": 1250,
        "temperature": 29
    }
])

prediction = model.predict(sample)

print(
    f"Predicted Yield: {prediction[0]:.0f} kg/hectare"
)