import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Multi-Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load BOTH models into memory
classifier = joblib.load("classifier_model.joblib")
regressor = joblib.load("regressor_model.joblib")

class HouseFeatures(BaseModel):
    sqft: float
    bedrooms: int
    age: float

# Endpoint 1: Predict Price
@app.post("/predict/price")
def predict_price(data: HouseFeatures):
    input_data = [[data.sqft, data.bedrooms, data.age]]
    predicted_price = regressor.predict(input_data)[0]
    return {"estimated_price": round(predicted_price, 2)}

# Endpoint 2: Predict Category
@app.post("/predict/category")
def predict_category(data: HouseFeatures):
    input_data = [[data.sqft, data.bedrooms, data.age]]
    prediction = classifier.predict(input_data)[0]
    probabilities = classifier.predict_proba(input_data)[0].tolist()
    
    result = "High Value" if prediction == 1 else "Low Value"
    return {
        "category": result,
        "confidence": round(max(probabilities) * 100, 2)
    }