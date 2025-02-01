# 🐟 Feeding Optimization System

> The Feeding Optimization System is a tool designed to help aquaculture farmers optimize feeding practices by analyzing feeding behavior, predicting optimal feed amounts, and generating actionable recommendations using machine learning and natural language generation.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Folder Structure](#folder-structure)
- [Usage](#usage)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Project Overview

This project uses computer vision, machine learning, and natural language generation (LLM) to:

- Detect fish feeding activity using synthetic images
- Predict the optimal feed amount based on feeding activity scores and water quality parameters
- Generate actionable recommendations for farmers using OpenAI's GPT

## ✨ Features

- **Feeding Activity Detection**: Analyze synthetic fish images to calculate feeding activity scores
- **Optimal Feed Prediction**: Use a machine learning model to predict the optimal feed amount
- **Dynamic Recommendations**: Generate detailed, natural-language recommendations using OpenAI's GPT
- **Real-Time Dashboard**: Visualize real-time data, historical trends, and recommendations in a lightweight web interface

## 🛠 Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- Pip (Python package manager)
- OpenAI API Key (sign up at [OpenAI](https://openai.com))
- A code editor (e.g., VS Code, PyCharm)

## 📥 Installation

### Step 1: Clone the Repository


git clone https://github.com/yourusername/feeding-optimization-system.git
cd feeding-optimization-system


### Step 2: Install Dependencies

Install the required Python libraries:


pip install -r requirements.txt


If `requirements.txt` is not provided, install the dependencies manually:


pip install flask openai python-dotenv pandas scikit-learn joblib ultralytics matplotlib chart.js


### Step 3: Set Up Environment Variables

1. Create a `.env` file in the root directory:
   
   touch .env
   

2. Add your OpenAI API key to the `.env` file:
   
   OPENAI_API_KEY=your_openai_api_key_here
   

3. Add `.env` to your `.gitignore` file to prevent it from being committed:
   
   echo ".env" >> .gitignore
   

### Step 4: Prepare Data

- Place your synthetic fish images in the `data/synthetic/fish_images/` folder
- Ensure `water_data.csv` and `feeding_events.csv` are available in the `data/` folder

## 🚀 Usage

### Step 1: Train the Machine Learning Model

Run the `train_model.py` script to train the Random Forest Regressor:


python src/train_model.py


The trained model will be saved as `feeding_model.joblib` in the `models/` folder.

### Step 2: Run the Flask Backend

Start the Flask server:


python app.py


The server will start running at `http://127.0.0.1:5000/`.

### Step 3: Access the Dashboard

Open your browser and navigate to:


http://127.0.0.1:5000/


The dashboard will display:
- Real-time feeding activity scores and water quality data
- Historical feeding activity trends
- Optimal feed amount and recommendations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

📊 **Feeding Optimization System** | Developed with ❤️ for Aquaculture
