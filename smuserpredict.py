import os
import joblib
import numpy as np

import warnings
warnings.filterwarnings("ignore")



def load_model(stock_name, models_dir):
    """
    Loads the trained model for a given stock name.

    Args:
        stock_name: Name of the stock (without file extension)
        models_dir: Directory where models are saved

    Returns:
        model: Loaded Linear Regression model
    """
    model_path = os.path.join(models_dir, f"{stock_name}_SMmodel.joblib")

    if not os.path.exists(model_path):
        print(f"[Error] Model not found for stock: {stock_name}")
        return None

    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


def predict_stock_price(stock_name, close, volume, models_dir):
    """
    Predicts the next day's close price for a given stock using its trained model.

    Args:
        stock_name: Name of the stock (without file extension)
        close: Close price for the current day
        volume: Trading volume for the current day
        models_dir: Directory where models are saved

    Returns:
        Predicted close price for the next day or None if prediction fails
    """
    model = load_model(stock_name, models_dir)

    if model is None:
        return None

    # Prepare the input feature array
    X_input = np.array([[close, volume]])  # 2D array for sklearn model
    predicted_close = model.predict(X_input)[0]

    print(f"PREDICTED CLOSE PRICE FOR {stock_name} (NEXT DAY): {predicted_close:.2f}")
    return predicted_close


if __name__ == "__main__":
    models_directory = r"C:\Users\GOWTHAM.S\PycharmProjects\StockMarket\StockMarketAnalysis\models"

    # Get user input
    stock_name = input("Enter the stock name (e.g., aapl.us): ").strip()
    try:
        current_close_price = float(input("Enter the current close price: "))
        current_volume = int(input("Enter the current trading volume: "))
    except ValueError:
        print("[ERROR] Invalid input. Please enter numerical values for price and volume.")
        exit()

    # Make prediction
    predict_stock_price(stock_name, current_close_price, current_volume, models_directory)
