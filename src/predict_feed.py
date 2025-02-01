import joblib
import pandas as pd

# Load the trained model
def load_model():
    model = joblib.load("../models/feeding_model.joblib")
    return model

# Predict the optimal feed amount
def predict_feed_amount(activity_score, temperature, oxygen, ph, turbidity):
    # Load the trained model
    model = load_model()

    # Prepare input data
    input_data = pd.DataFrame({
        'activity_score': [activity_score],
        'temperature': [temperature],
        'oxygen': [oxygen],
        'ph': [ph],
        'turbidity': [turbidity]
    })

    # Predict the feed amount
    predicted_feed_amount = model.predict(input_data)
    return predicted_feed_amount[0]

if __name__ == "__main__":
    # Example input data
    activity_score = 0.9
    temperature = 25.5
    oxygen = 6.8
    ph = 7.2
    turbidity = 15.0

    # Predict the feed amount
    feed_amount = predict_feed_amount(activity_score, temperature, oxygen, ph, turbidity)
    print(f"Predicted Feed Amount: {feed_amount:.2f} kg")