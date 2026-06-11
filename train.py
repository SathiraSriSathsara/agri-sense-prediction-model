import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load data
df = pd.read_csv("crop_data.csv")

# Features
X = df.drop("yield", axis=1)
y = df["yield"]

# Categorical columns
categorical_features = ["district", "crop", "season"]

# Numerical columns
numerical_features = [
    "ph",
    "nitrogen",
    "phosphorus",
    "potassium",
    "rainfall",
    "temperature"
]

# Encode text columns
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# Build model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    ))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Test
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print(f"Mean Absolute Error: {mae:.2f}")

# Save model
joblib.dump(model, "crop_model.pkl")

print("Model saved as crop_model.pkl")