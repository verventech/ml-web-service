# 🏡 California Housing ML Predictor

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A complete, end-to-end machine learning web application that predicts housing prices in California. This project demonstrates how to train, serialize, and serve classical Machine Learning models (Random Forest) via a REST API, consumed by a lightweight frontend web UI.

---

## 📖 About the Project

This project uses the famous **California Housing Dataset** to build two distinct machine learning models:
1. **Random Forest Regressor:** Predicts the exact estimated dollar value of a property based on block group demographics.
2. **Random Forest Classifier:** Predicts whether a property's value sits "Above" or "Below" the statewide median price, providing a confidence percentage.

The system is designed with a decoupled architecture:
*   **Backend:** A Python-based `FastAPI` server that loads the serialized `.joblib` models into memory and serves predictions via HTTP endpoints.
*   **Frontend:** A vanilla HTML/JS web page that collects user input and fetches predictions from the backend without page reloads.

---

## 📂 Project Structure

```text
ml_web_service/
│
├── train_both.py              # ML script to train and save both models
├── app.py                     # FastAPI backend application
├── index.html                 # Frontend Web UI
├── classifier_model.joblib    # Generated: Saved Classifier model weights (Not in version control)
├── regressor_model.joblib     # Generated: Saved Regressor model weights (Not in version control)
└── README.md                  # This documentation file
```

---

## 🚀 Getting Started

Follow these instructions to set up and run the project on a local Ubuntu/Linux environment.

### 1. System Requirements & Environment Setup
Ensure you have Python 3 installed. Open your terminal and create a virtual environment:

```bash
# Update package lists and install venv if needed
sudo apt update
sudo apt install -y python3-venv python3-pip

# Create project directory
mkdir ml_web_service && cd ml_web_service

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install fastapi uvicorn scikit-learn joblib pydantic numpy
```

### 2. Train the Models
Before starting the API, you must train the models and generate the `.joblib` files. Run the training script:

```bash
python train_both.py
```
*Expected Output: The script will download the dataset, perform a train/test split, train the models, print accuracy/R² metrics, and save `classifier_model.joblib` and `regressor_model.joblib` to your directory.*

### 3. Start the API Server
Start the FastAPI backend using `uvicorn`. The `--reload` flag enables auto-reloading during development.

```bash
uvicorn app:app --reload --port 8000
```
*Your API is now running locally at `http://127.0.0.1:8000`.*

### 4. Open the Web UI
Open the `index.html` file in your preferred web browser:
*   **Linux/Mac:** `open index.html`
*   **Alternatively:** Just drag and drop the `index.html` file into Chrome or Firefox.

Enter the demographic data into the input fields and click the prediction buttons to see the machine learning models in action!

---

## 📡 API Endpoints Documentation

If you want to interact with the models programmatically (e.g., via `curl` or Postman), you can use the following endpoints. 

**Base URL:** `http://127.0.0.1:8000`

### 1. Predict Price (Regression)
*   **URL:** `/predict/price`
*   **Method:** `POST`
*   **Payload:** JSON object matching the `HouseFeatures` schema.
*   **Success Response:** `{"estimated_price": 450500.25}`

### 2. Predict Category (Classification)
*   **URL:** `/predict/category`
*   **Method:** `POST`
*   **Payload:** JSON object matching the `HouseFeatures` schema.
*   **Success Response:** `{"category": "Above California Median", "confidence": 85.5}`

### Interactive API Docs
FastAPI automatically generates interactive Swagger documentation. While the server is running, visit:
👉 **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---

## 🛠️ Built With

*   [Scikit-Learn](https://scikit-learn.org/) - Machine learning library (RandomForest, datasets, metrics)
*   [FastAPI](https://fastapi.tiangolo.com/) - High-performance web framework for APIs
*   [Uvicorn](https://www.uvicorn.org/) - ASGI web server implementation for Python
*   [Joblib](https://joblib.readthedocs.io/) - Serialization library for persisting models to disk
