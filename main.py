
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import joblib
import numpy as np

app = FastAPI()

# Allow CORS for your frontend (adjust the origin as necessary)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory where models are stored
MODELS_DIR = r"C:\Users\GOWTHAM.S\PycharmProjects\StockMarket\StockMarketAnalysis\models"


def load_model(stock_name):
    """
    Loads the trained model for a given stock name.

    Args:
        stock_name: Name of the stock (without file extension)

    Returns:
        model: Loaded Linear Regression model or None if not found
    """
    model_path = os.path.join(MODELS_DIR, f"{stock_name}_SMmodel.joblib")

    if not os.path.exists(model_path):
        return None

    return joblib.load(model_path)


@app.post("/predict/")
async def predict(stock_name: str = Form(...), close: float = Form(...), volume: int = Form(...)):
    """
    API endpoint to predict the next day's close price for a given stock.

    Args:
        stock_name: Name of the stock (from the form)
        close: Close price for the current day
        volume: Trading volume for the current day

    Returns:
        JSON response with predicted close price or an error message
    """
    model = load_model(stock_name)

    if model is None:
        return JSONResponse({"error": f"Model not found for stock: {stock_name}"}, status_code=404)

    # Prepare the input feature array
    X_input = np.array([[close, volume]])  # 2D array for sklearn model
    predicted_close = model.predict(X_input)[0]

    return {"predicted_close": round(predicted_close, 2)}
