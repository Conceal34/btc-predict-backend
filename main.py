from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import joblib
import os, warnings
warnings.filterwarnings("ignore")

from keras.models import load_model

from dotenv import load_dotenv
load_dotenv()

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")



app = FastAPI(title="PriceOracle API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["FRONTEND_URL"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_TICKERS = ["BTC-USD", "ETH-USD"]
BASE_DAYS = 100

_models  = {}
_scalers = {}

def load_models():
    for ticker in SUPPORTED_TICKERS:
        model_path  = f"models/{ticker}_lstm.keras"
        scaler_path = f"models/{ticker}_scaler.pkl"

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            _models[ticker]  = load_model(model_path)
            _scalers[ticker] = joblib.load(scaler_path)
            print(f"[startup] Loaded model and scaler for {ticker}")
        else:
            print(f"[startup] WARNING: Files missing for {ticker}. Run model_training.ipynb first.")

load_models()


# fetch closing price data
def get_closing(ticker: str, years: int = 5) -> pd.DataFrame:
    end   = datetime.now()
    start = datetime(end.year - years, end.month, end.day)
    data  = yf.download(ticker, start=start, end=end, progress=False)

    if data.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data[["Close"]].dropna()


# Request / Response TYPES/ Pydantic MODELS
class ForecastRequest(BaseModel):
    ticker:        str = "BTC-USD"
    forecast_days: int = 10


class TickerInfoResponse(BaseModel):
    ticker:        str
    name:          str
    current_price: float
    change_pct:    float
    high_52w:      float
    low_52w:       float
    volume:        float


# Routes 
@app.get("/")
def root():
    return {"message": "PriceOracle API is running"}

