import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Step 1: Load and merge datasets
def merge_datasets():
    # Load datasets
    feeding_events = pd.read_csv("../data/feeding_events.csv")
    water_data = pd.read_csv("../data/synthetic/water_data.csv")

    # Remove 'Unnamed: 0' column if it exists in water_data
    if 'Unnamed: 0' in water_data.columns:
        water_data = water_data.drop(columns=['Unnamed: 0'])

    # Convert timestamps to datetime objects
    feeding_events['timestamp'] = pd.to_datetime(feeding_events['timestamp'])
    water_data.index = pd.to_datetime(water_data.index)

    # Merge datasets on timestamp using pd.merge_asof
    merged_data = pd.merge_asof(
        feeding_events.sort_values('timestamp'),
        water_data,
        left_on='timestamp',
        right_index=True,
        direction='nearest'
    )

    # Drop duplicate rows based on timestamp
    merged_data = merged_data.drop_duplicates(subset=['timestamp'])

    # Save merged dataset
    merged_data.to_csv("../data/merged_dataset.csv", index=False)
    print("Merged dataset saved to ../data/merged_dataset.csv")
    return merged_data

# Step 2: Train the model
def train_model():
    # Load merged dataset
    data = pd.read_csv("../data/merged_dataset.csv")

    # Check for missing values
    if data.isnull().values.any():
        print("Warning: Missing values found in the dataset. Dropping rows with missing values.")
        data = data.dropna()

    # Features and target
    X = data[['activity_score', 'temperature', 'oxygen', 'ph', 'turbidity']]  # Input features
    y = data['feed_amount']  # Target variable (optimal feed amount)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Save the trained model
    joblib.dump(model, "../models/feeding_model.joblib")
    print("Model saved to ../models/feeding_model.joblib")

if __name__ == "__main__":
    # Merge datasets
    merge_datasets()

    # Train the model
    train_model()