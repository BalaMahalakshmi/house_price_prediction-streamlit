
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# --- Configuration ---
FILE_PATH = "data.csv"
TARGET_COLUMN = 'price'
# Columns used for training. I added bathrooms/floors for better performance.
FEATURE_COLUMNS = ['bedrooms', 'sqft_lot', 'sqft_living', 'bathrooms', 'floors']
PKL_FILE_NAME = 'house_price_model.pkl'

# --- 1. Load, Select, and Clean Data ---
try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}")
    exit()

# Select only the required columns
df_selected = df[[TARGET_COLUMN] + FEATURE_COLUMNS].copy()

# Drop rows with any missing values in the selected columns
df_selected.dropna(inplace=True)

X = df_selected[FEATURE_COLUMNS]
y = df_selected[TARGET_COLUMN]

# --- 2. Feature Engineering and Target Transformation ---

# Feature Engineering: SqFt Per Bedroom (Interaction Term)
# X['SqFt_Per_Bedroom'] = X['sqft_living'] / X['bedrooms']
# Clean up infinite/NaN values created by division by zero (if any house has 0 bedrooms)
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(inplace=True)

# Align y after dropping rows due to engineering
y = y[X.index]

# Target Transformation: Log transform the price (Essential for high R2)
y_log = np.log1p(y)

# --- 3. Split Data ---
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

# --- 4. Define Preprocessing Pipeline ---
numerical_features = X.columns.tolist() # All features are now numerical

preprocessor = ColumnTransformer(
    transformers=[
        # Scaling numerical features
        ('num', StandardScaler(), numerical_features)
    ],
    remainder='passthrough'
)

# --- 5. Full ML Pipeline and Training (Random Forest Regressor) ---

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # Using robust hyperparameters
    ('regressor', RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1))
])

print("\n--- Starting Model Training ---")
model_pipeline.fit(X_train, y_train_log)
print("Model Training Complete.")

# --- 6. Evaluation (Optional, but good for diagnostics) ---
y_pred_log = model_pipeline.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test = np.expm1(y_test_log)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Test Set R2 Score: {r2:.4f}")
print(f"Test Set MAE: ${mae:,.2f}")

# --- 7. Save the Model Pipeline to PKL ---
joblib.dump(model_pipeline, PKL_FILE_NAME)
print(f"\nModel pipeline successfully saved as '{PKL_FILE_NAME}'")