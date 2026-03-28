```markdown
# PriceOracle — BTC Predict Backend

A FastAPI backend that serves **LSTM-based price predictions** for Bitcoin and Ethereum. It pulls live market data from Yahoo Finance, runs it through pre-trained Keras/TensorFlow models, and returns historical context, test-set accuracy, and a configurable future forecast — all in one JSON response.

> **Disclaimer:** This project is for educational purposes only. Nothing here constitutes financial advice.

---

# Features

- 📈 **Live market data** fetched in real-time via `yfinance`
- 🧠 **LSTM neural network** predictions for `BTC-USD` and `ETH-USD`
- 🔁 **Autoregressive forecasting** — each predicted day feeds into the next
- 📊 **Test-set evaluation** with MAE and RMSE metrics on every forecast call
- 📉 **Moving averages** (MA100, MA365) via the history endpoint
- 🌐 **CORS-configured** for seamless frontend integration

---

## 🗂️ Project Structure

```
btc-predict-backend/
├── main.py                  # FastAPI app — all routes & prediction logic
├── model_training.ipynb     # Jupyter notebook to train & save models
├── requirements.txt
├── runtime.txt              # python-3.12.12
└── models/                  # ⚠️ Not committed — generate with the notebook
    ├── BTC-USD_lstm.keras
    ├── BTC-USD_scaler.pkl
    ├── ETH-USD_lstm.keras
    └── ETH-USD_scaler.pkl
```



## Setup
```
**Prerequisites:** Python 3.12, pip

```bash
```
# Clone & enter the repo
git clone https://github.com/Conceal34/btc-predict-backend.git
cd btc-predict-backend

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up your environment variables
echo "FRONTEND_URL=http://localhost:3000" > .env

# Train the models first (required!)
jupyter notebook model_training.ipynb

# Start the server
uvicorn main:app --reload --port 8000


> Interactive API docs available at **`http://localhost:8000/docs`**

---

## Environment Variables

Create a `.env` file in the project root:

```env
FRONTEND_URL=http://localhost:3000
```

|------------------------------------------------------------------------------|
| Variable       | Default                 | Description                       |
|----------------|-------------------------|-----------------------------------|
| `FRONTEND_URL` | `http://localhost:3000` | Origin allowed by CORS middleware |
|------------------------------------------------------------------------------|
---




# API Endpoints

### `GET /`
Health check.
```json
{ "message": "PriceOracle API is running" }
```

---

### `GET /ticker/{symbol}`
Current market snapshot for a supported ticker.

**Supported:** `BTC-USD`, `ETH-USD`

```json
{
  "ticker": "BTC-USD",
  "name": "Bitcoin USD",
  "current_price": 68420.50,
  "change_pct": 2.31,
  "high_52w": 73750.00,
  "low_52w": 38500.00,
  "volume": 29847362048.0
}
```

---

### `POST /forecast`
Main prediction endpoint. Returns test accuracy, future projections, and historical context.

**Request:**
```json
{
  "ticker": "BTC-USD",
  "forecast_days": 10
}
```

**Response:**
```json
{
  "ticker": "BTC-USD",
  "historical": { "dates": ["..."], "prices": [57000.0] },
  "test":       { "dates": ["..."], "actual": [92000.0], "predicted": [91200.0] },
  "forecast":   { "dates": ["..."], "prices": [85000.0] },
  "metrics":    { "mae": 1234.56, "rmse": 1987.43 }
}
```

### `GET /history/{symbol}?years=5`
Full closing price history with 100-day and 365-day moving averages.

```json
{
  "dates": ["2020-01-01", "..."],
  "close": [7200.0],
  "ma100": [null, "...", 8500.0],
  "ma365": [null, "...", 9100.0]
}
```


# How the Prediction Works

1. **Fetches** 5 years of daily closing prices from Yahoo Finance
2. **Normalizes** data using the pre-fitted `MinMaxScaler` from training
3. **Builds sequences** of 100 consecutive days as model input windows
4. **Splits** 90/10 for test evaluation — runs the model on the held-out 10%
5. **Rolls forward** autoregressively for future days — each output becomes the next input
6. **Inverse-transforms** all predictions back to USD and computes MAE + RMSE

> The scaler is saved at training time and must match what the model was trained on. Never refit it at inference.


# Dependencies
|------------------------------|
| Package             |Version |
|---------------------|--------|
| `fastapi`           | 0.115.0|
| `uvicorn[standard]` | 0.30.0 |
| `pydantic`          | 2.7.0  |
| `yfinance`          | ≥1.2.0 |
| `pandas`            | ≥1.3.0 |
| `numpy`             | ≥1.16.5|
| `scikit-learn`      | ≥1.0.0 |
| `keras`             | ≥2.10.0|
| `tensorflow`        | ≥2.10.0|
| `python-multipart`  | 0.0.9  |
|------------------------------|

# Deployment

- `runtime.txt` declares `python-3.12.12` for platforms like **Heroku** or **Railway**
- Model files (`*.keras`, `*.pkl`) must exist in `models/` at startup — missing files log a warning and cause `503` errors on affected endpoints
- Set `FRONTEND_URL` to your production domain in environment config
- `/forecast` makes a live `yfinance` call on every request — consider caching responses for high-traffic deployments
