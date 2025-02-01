from flask import Flask, render_template, jsonify
import joblib
import pandas as pd
from src.generate_recommendations import generate_recommendation

app = Flask(__name__)

# Load the trained model
model = joblib.load("models/feeding_model.joblib")

# Simulated real-time data (replace with actual data from sensors)
def get_real_time_data():
    return {
        "activity_score": 0.9,
        "temperature": 50,
        "oxygen": 6.8,
        "ph": 7.2,
        "turbidity": 15.0
    }

# Simulated historical feeding activity data
def get_historical_data():
    return {
        "timestamps": ["2024-01-01 08:00", "2024-01-01 08:10", "2024-01-01 08:20", "2024-01-01 08:30"],
        "activity_scores": [0.85, 0.90, 0.88, 0.92]
    }

# Route for the dashboard
@app.route("/")
def dashboard():
    return render_template("dashboard.html")

# API endpoint to fetch real-time data
@app.route("/api/data", methods=["GET"])
def api_data():
    # Get real-time data
    data = get_real_time_data()

    # Predict feed amount
    input_data = pd.DataFrame([[
        data["activity_score"],
        data["temperature"],
        data["oxygen"],
        data["ph"],
        data["turbidity"]
    ]], columns=["activity_score", "temperature", "oxygen", "ph", "turbidity"])
    feed_amount = model.predict(input_data)[0]

    # Generate recommendation (mock for now)
       # Generate recommendation using the imported function
    recommendation = generate_recommendation(
        activity_score=data["activity_score"],
        temperature=data["temperature"],
        oxygen=data["oxygen"],
        ph=data["ph"],
        turbidity=data["turbidity"],
        feed_amount=feed_amount
    )

    # Return data as JSON
    return jsonify({
        "activity_score": data["activity_score"],
        "temperature": data["temperature"],
        "oxygen": data["oxygen"],
        "ph": data["ph"],
        "turbidity": data["turbidity"],
        "feed_amount": feed_amount,
        "recommendation": recommendation
    })

# API endpoint to fetch historical data for the chart
@app.route("/api/historical-data", methods=["GET"])
def api_historical_data():
    historical_data = get_historical_data()
    return jsonify(historical_data)

if __name__ == "__main__":
    app.run(debug=True)