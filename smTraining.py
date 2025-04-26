import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def load_stock_data(file_path):
    """
    Loads stock data from a CSV-like file and preprocesses it.

    Args:
        file_path: Path to the stock data file.

    Returns:
        X: Features (Close, Volume)
        y: Target (Close price of the next day)
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"[Error] File not found: {file_path}")
        return None, None
    except pd.errors.EmptyDataError:
        print(f"[Error] File is empty: {file_path}")
        return None, None
    except Exception as e:
        print(f"[Error] Failed to load file {file_path}: {e}")
        return None, None

    required_columns = {'Date', 'Close', 'Volume'}
    if not required_columns.issubset(df.columns):
        print(f"[Error] Missing required columns in {file_path}. Expected: {required_columns}")
        return None, None

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Convert Date column to datetime
    df['Close_Tomorrow'] = df['Close'].shift(-1)  # Target: next day's Close price
    df = df.dropna()  # Drop rows with NaN values

    X = df[['Close', 'Volume']]
    y = df['Close_Tomorrow']
    return X, y


def train_and_save_model(X, y, model_name, models_dir):
    """
    Trains a Linear Regression model and saves it to a file in the specified models directory.

    Args:
        X: Features for training
        y: Target values for training
        model_name: Name of the model file
        models_dir: Directory to save the model

    Returns:
        None
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluate model performance
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Model trained for {model_name}. MSE: {mse:.4f}, R²: {r2:.4f}")

        # Save the model in the specified directory
        os.makedirs(models_dir, exist_ok=True)  # Create models directory if it doesn't exist
        save_path = os.path.join(models_dir, model_name)
        joblib.dump(model, save_path)
        print(f"Model saved to {save_path}")

    except Exception as e:
        print(f"[Error] Failed to train and save model: {e}")


def process_directory(root_dir, sub_dir, models_dir):
    """
    Processes all CSV-like files in a given directory and trains models.

    Args:
        root_dir: Root directory containing subdirectories
        sub_dir: Subdirectory (e.g., 'Stocks' or 'ETFs') to process
        models_dir: Directory to save the models

    Returns:
        None
    """
    dir_path = os.path.join(root_dir, sub_dir)

    if not os.path.exists(dir_path):
        print(f"[Warning] Directory not found: {dir_path}")
        return

    # List all files in the directory for debugging
    print(f"Files in {dir_path}: {os.listdir(dir_path)}")

    # Look for files with .csv or .us extensions
    csv_files = [f for f in os.listdir(dir_path) if f.endswith((".csv", ".us", ".us.txt"))]

    if not csv_files:
        print(f"[Warning] No CSV-like files found in {dir_path}")
        return

    for filename in csv_files:
        file_path = os.path.join(dir_path, filename)
        print(f"Processing file: {file_path}")

        X, y = load_stock_data(file_path)

        if X is not None and y is not None:
            model_name = f"{os.path.splitext(filename)[0]}_SMmodel.joblib"
            train_and_save_model(X, y, model_name, models_dir)


if __name__ == "__main__":
    root_directory = r"E:\StockMarketDataset"
    models_directory = r"C:\Users\GOWTHAM.S\PycharmProjects\StockMarketAnalysis\models"

    # Process stocks
    print("Processing Stocks...")
    process_directory(root_directory, "Stocks", models_directory)

    # Process ETFs
    print("\nProcessing ETFs...")
    process_directory(root_directory, "ETFs", models_directory)
