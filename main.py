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
    allow_origins=[FRONTEND_URL],
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


@app.get("/ticker/{symbol}", response_model=TickerInfoResponse)
def get_ticker_info(symbol: str):
    symbol = symbol.upper()
    if symbol not in SUPPORTED_TICKERS:
        raise HTTPException(status_code=400, detail=f"Unsupported ticker. Choose from: {SUPPORTED_TICKERS}")

    try:
        info = yf.Ticker(symbol).info
        hist = yf.download(symbol, period="8d", progress=False)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        current = float(hist["Close"].iloc[-1])
        prev    = float(hist["Close"].iloc[-8]) if len(hist) >= 8 else current
        chg_pct = (current - prev) / prev * 100

        return TickerInfoResponse(
            ticker        = symbol,
            name          = info.get("shortName", symbol),
            current_price = current,
            change_pct    = round(chg_pct, 2),
            high_52w      = float(info.get("fiftyTwoWeekHigh", 0)),
            low_52w       = float(info.get("fiftyTwoWeekLow",  0)),
            volume        = float(info.get("volume", 0) or 0),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/forecast")
def run_forecast(req: ForecastRequest):
    ticker = req.ticker.upper()

    if ticker not in SUPPORTED_TICKERS:
        raise HTTPException(status_code=400, detail=f"Unsupported ticker. Choose from: {SUPPORTED_TICKERS}")

    if ticker not in _models or ticker not in _scalers:
        raise HTTPException(status_code=503, detail=f"Model or scaler for {ticker} is not loaded. Run model_training.ipynb first.")

    try:
        model   = _models[ticker]
        scaler  = _scalers[ticker]
        closing = get_closing(ticker)
        scaled  = scaler.transform(closing)

        # ── Test set predictions ──────────────────────────────────────────────
        x_data, y_data = [], []
        for i in range(BASE_DAYS, len(scaled)):
            x_data.append(scaled[i - BASE_DAYS:i])
            y_data.append(scaled[i])
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        split    = int(len(x_data) * 0.9)
        x_test   = x_data[split:]
        y_test   = y_data[split:]

        preds      = model.predict(x_test, verbose=0)
        inv_preds  = scaler.inverse_transform(preds).flatten().tolist()
        inv_actual = scaler.inverse_transform(y_test).flatten().tolist()

        test_dates = [
            closing.index[split + BASE_DAYS + i].strftime("%Y-%m-%d")
            for i in range(len(inv_actual))
        ]

        # ── Future forecast ───────────────────────────────────────────────────
        last_seq      = scaled[-BASE_DAYS:].reshape(1, -1, 1)
        future_prices = []

        for _ in range(req.forecast_days):
            nxt = model.predict(last_seq, verbose=0)
            future_prices.append(float(scaler.inverse_transform(nxt)[0, 0]))
            last_seq = np.append(last_seq[:, 1:, :], nxt.reshape(1, 1, 1), axis=1)

        end = datetime.now()
        future_dates = [
            (end + timedelta(days=i + 1)).strftime("%Y-%m-%d")
            for i in range(req.forecast_days)
        ]

        # ── Historical (last 90 days for chart context) ───────────────────────
        hist_prices = closing["Close"].values[-90:].tolist()
        hist_dates  = [d.strftime("%Y-%m-%d") for d in closing.index[-90:]]

        # ── Metrics ───────────────────────────────────────────────────────────
        mae  = float(np.mean(np.abs(np.array(inv_preds) - np.array(inv_actual))))
        rmse = float(np.sqrt(np.mean((np.array(inv_preds) - np.array(inv_actual)) ** 2)))

        return {
            "ticker":     ticker,
            "historical": {"dates": hist_dates,   "prices": hist_prices},
            "test":       {"dates": test_dates,    "actual": inv_actual, "predicted": inv_preds},
            "forecast":   {"dates": future_dates,  "prices": future_prices},
            "metrics":    {"mae": round(mae, 2),   "rmse": round(rmse, 2)},
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/history/{symbol}")
def get_history(symbol: str, years: int = 5):
    symbol = symbol.upper()

    if symbol not in SUPPORTED_TICKERS:
        raise HTTPException(status_code=400, detail=f"Unsupported ticker. Choose from: {SUPPORTED_TICKERS}")

    try:
        end   = datetime.now()
        start = datetime(end.year - years, end.month, end.day)
        data  = yf.download(symbol, start=start, end=end, progress=False)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        closing = data["Close"].dropna()
        ma100   = closing.rolling(100).mean()
        ma365   = closing.rolling(365).mean()

        return {
            "dates": [d.strftime("%Y-%m-%d") for d in closing.index],
            "close": [float(v) if not np.isnan(v) else None for v in closing.values],
            "ma100": [float(v) if not np.isnan(v) else None for v in ma100.values],
            "ma365": [float(v) if not np.isnan(v) else None for v in ma365.values],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
