# Quant Prediction Market Bot

A lightweight research bot for prediction markets using the DMDO framework:

**Data → Model → Decision → Output**

## Current Status

The project currently includes:
- Mock market data ingestion
- Bayesian probability updates
- Expected value signal logic
- Kelly-style risk sizing
- Simple backtesting
- Telegram alert scaffolding

## Project Structure

```text
quant-bot/
├── app/
│   ├── main.py
│   ├── data_client.py
│   ├── bayes.py
│   ├── signals.py
│   ├── risk.py
│   ├── alerts.py
│   ├── state.py
│   └── backtest.py
├── data/
├── logs/
├── requirements.txt
└── README.md
```

## Run the Bot

```bash
python -m app.main
```

## Run the Backtest

```bash
python -m app.backtest
```

## DMDO Roadmap

1. **Data** — ingest market data cleanly
2. **Model** — estimate true probabilities
3. **Decision** — compute EV and size trades
4. **Output** — alerts, logs, dashboards, automation

## Disclaimer

This project is for research and educational purposes only. It is not financial advice.
