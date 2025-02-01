from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_recommendation(activity_score, temperature, oxygen, ph, turbidity, feed_amount):
    # Create a prompt for the LLM
    prompt = f"""
    Based on the following conditions:
    - Feeding Activity Score: {activity_score}
    - Water Temperature: {temperature}°C
    - Dissolved Oxygen: {oxygen} mg/L
    - pH Level: {ph}
    - Turbidity: {turbidity} NTU

    The optimal feed amount is {feed_amount:.2f} kg. Provide a recommendation for the farmer.
    """

    # Call the OpenAI API using the new client format
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the recommendation using the new response format
    recommendation = completion.choices[0].message.content
    return recommendation

if __name__ == "__main__":
    # Example input data
    activity_score = 0.9
    temperature = 25.5
    oxygen = 6.8
    ph = 7.2
    turbidity = 15.0
    feed_amount = 1.47

    # Generate a recommendation
    recommendation = generate_recommendation(activity_score, temperature, oxygen, ph, turbidity, feed_amount)
    print("Recommendation:")
    print(recommendation)
