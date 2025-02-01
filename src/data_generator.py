import pandas as pd
import numpy as np
import cv2
import os
from faker import Faker
import matplotlib .pyplot as plt
import argparse
from PIL import Image


# Function to generate water data
def generate_water_data(days=30):
    fake = Faker()
    dates = pd.date_range(start="2024-01-01", periods=days, freq="D")
    
    # Seasonal sine wave for temperature (annual cycle)
    temp_base = 18  # Mean temperature
    temp_seasonal = 5 * np.sin(2 * np.pi * np.arange(days) / 365)  # Annual cycle
    temperature = temp_base + temp_seasonal + np.random.normal(0, 1, days)
    
    # Oxygen depends on temperature (inverse relationship)
    oxygen = np.clip(10 - 0.2 * temperature + np.random.normal(0, 0.3, days), 4, 8)
    
    # pH is stable but slightly fluctuates
    ph = np.clip(np.random.normal(7.5, 0.2, days), 6.5, 8.5)
    
    # Turbidity increases with feeding activity
    feed_amount = np.clip(np.random.normal(1.2, 0.3, days), 0.5, 2.0)
    turbidity = 1 + 2 * feed_amount + np.random.normal(0, 0.5, days)
    
    data = {
        "temperature": temperature,
        "oxygen": oxygen,
        "ph": ph,
        "turbidity": turbidity,
        "feed_amount": feed_amount  # Add feed_amount as a new column
    }
    df = pd.DataFrame(data, index=dates)
    df.to_csv("data/synthetic/water_data.csv")
    print("Water data generated and saved to data/synthetic/water_data.csv")




def generate_fish_images(num_images=100, img_size=(300, 300), num_fish_range=(1, 5)):
    """
    Generate synthetic fish images with multiple fish per image.
    Simulates clustering behavior for feeding analysis.
    """
    os.makedirs("data/synthetic/fish_images", exist_ok=True)
    
    for i in range(num_images):
        img = np.ones((*img_size, 3), dtype=np.uint8) * 255  # White background
        
        # Randomly determine the number of fish in the image
        num_fish = np.random.randint(*num_fish_range)
        
        # Simulate clustering behavior
        if np.random.rand() < 0.5:  # 50% chance of clustering
            cluster_center = (np.random.randint(50, 250), np.random.randint(50, 250))
            for _ in range(num_fish):
                x = cluster_center[0] + np.random.randint(-30, 30)
                y = cluster_center[1] + np.random.randint(-30, 30)
                x = np.clip(x, 0, img_size[0] - 10)  # Ensure fish stays within bounds
                y = np.clip(y, 0, img_size[1] - 10)
                
                # Draw a black circle for the fish
                cv2.circle(img, (x, y), 10, (0, 0, 0), -1)
        else:
            # Spread out fish randomly
            for _ in range(num_fish):
                x = np.random.randint(50, 250)
                y = np.random.randint(50, 250)
                
                # Draw a black circle for the fish
                cv2.circle(img, (x, y), 10, (0, 0, 0), -1)
        
        # Save image
        cv2.imwrite(f"data/synthetic/fish_images/fish_{i:04d}.png", img)
    
    print(f"{num_images} fish images generated and saved to data/synthetic/fish_images/")

# Function to generate feeding history
def generate_feeding_history(days=30):
    dates = pd.date_range(start="2024-01-01", periods=days, freq="D")
    feed_times = ["08:00", "16:00"]  # Morning and evening feedings
    
    feed_amounts = []
    growth_rates = []
    cumulative_growth = 0
    
    for day in range(days):
        daily_feed = 0
        for time in feed_times:
            amount = np.clip(np.random.normal(0.6, 0.1), 0.4, 1.0)  # Feed per session
            daily_feed += amount
        
        # Growth depends on cumulative feeding and water quality
        growth_rate = 0.15 * daily_feed + np.random.normal(0, 0.02)
        cumulative_growth += growth_rate
        
        feed_amounts.append(daily_feed)
        growth_rates.append(cumulative_growth)
    
    df = pd.DataFrame({
        "feed_amount": feed_amounts,
        "growth_rate": growth_rates
    }, index=dates)
    df.to_csv("data/synthetic/feeding_history.csv")
    print("Feeding history generated and saved to data/synthetic/feeding_history.csv")

# Main function to handle command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Data Generator for Aquaculture Simulation")
    parser.add_argument("--water", action="store_true", help="Generate synthetic water data")
    parser.add_argument("--fish", action="store_true", help="Generate synthetic fish images")
    parser.add_argument("--feeding", action="store_true", help="Generate synthetic feeding history")
    
    args = parser.parse_args()
    
    if args.water:
        generate_water_data()
    if args.fish:
        generate_fish_images()
    if args.feeding:
        generate_feeding_history()