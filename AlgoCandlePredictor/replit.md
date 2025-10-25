# AlgoTrader - Industry-Grade Trading Bot

## Project Overview

An AI-powered algorithmic trading platform with professional-grade machine learning models, Kelly Criterion position sizing, and risk management inspired by Renaissance Technologies, Two Sigma, and top quantitative hedge funds.

## Current Status

✅ **API Server**: Running on port 5000  
✅ **18 REST API Endpoints**: All functional  
✅ **ML Pipeline**: Hybrid XGBoost-LSTM ready for training  
✅ **Risk Management**: Industry-grade with Kelly Criterion support  
✅ **Demo Mode**: Virtual $100K paper trading account  
✅ **Backtesting**: Comprehensive performance metrics  

## Recent Changes (October 25, 2025)

### Completed Features

1. **ML Training Pipeline** (`models/train.py`)
   - Hybrid XGBoost-LSTM architecture
   - Two-stage training: Feature selection → Time-series prediction
   - Target accuracy: 70%+ directional accuracy
   - Automated model saving and versioning

2. **Technical Indicators** (`data/technical_indicators.py`)
   - 30+ indicators: RSI, MACD, Bollinger Bands, SMA/EMA, ATR, ADX
   - Price changes, volume analysis, candle patterns
   - Automated feature engineering pipeline

3. **Data Preprocessing** (`data/preprocess.py`)
   - Load and clean OHLCV data
   - Calculate technical indicators
   - Train/val/test split (70/15/15)
   - Normalization and sequence creation for LSTM
   - Feature selection integration

4. **Backtesting Engine** (`trading/backtester.py`)
   - Walk-forward backtesting
   - Sharpe ratio, max drawdown, win rate, profit factor
   - Equity curve tracking
   - Performance visualization ready

5. **Risk Management Enhancements** (`trading/risk_manager.py`)
   - Kelly Criterion position sizing framework
   - Half-Kelly conservative approach
   - Confidence-based position adjustment
   - ATR-based volatility adaptation

6. **Configuration** (`configs/config.yml`)
   - Demo mode with $100K virtual capital
   - Kelly Criterion parameters
   - ULTRACEMCO as primary trading symbol
   - Risk limits: 2% per trade, $5K daily loss limit

## Data Available

- **ULTRACEMCO**: 194,561 candles (5-minute, 2015-2025)
- **TVSMOTOR**: Full 5-minute historical data
- **UNITDSPR**: Full 5-minute historical data  
- **ZYDUSLIFE**: Full 5-minute historical data

All data stored in `data_files/` directory.

## Architecture

### ML Pipeline
```
Raw Data → Technical Indicators → XGBoost Feature Selection → 
→ LSTM Training → Model Evaluation → Save Models
```

### Trading Flow
```
Live Data → Prediction Service → Risk Manager → 
→ Trade Executor → Position Management
```

### API Stack
- **Framework**: FastAPI (async, high performance)
- **Server**: Uvicorn with hot reload
- **Documentation**: Auto-generated Swagger UI at /docs
- **Validation**: Pydantic models

## Key Files

- `api/main.py`: FastAPI application entry point
- `models/train.py`: ML model training script  
- `models/predict.py`: Prediction service
- `trading/executor.py`: Trade execution engine
- `trading/risk_manager.py`: Risk management with Kelly Criterion
- `trading/backtester.py`: Backtesting engine
- `data/technical_indicators.py`: Technical indicator calculations
- `data/preprocess.py`: Data preprocessing pipeline
- `configs/config.yml`: Trading configuration

## Next Steps

1. **Train Model**: Run `python models/train.py` to train on ULTRACEMCO data
2. **Test Predictions**: Use `/trading/predict` API endpoint
3. **Start Demo Trading**: Enable via config, start with $100K virtual
4. **Monitor Performance**: Track via `/system/metrics` endpoint
5. **Broker Integration**: Add real broker API (Zerodha/AngelOne)
6. **Production Deployment**: Scale when ready for live trading

## Research & Inspiration

### Top Quant Firms Studied
- **Renaissance Technologies**: Medallion Fund (39% annual returns, Sharpe 2+)
- **Two Sigma**: Machine learning + big data approach
- **Citadel**: Multi-strategy with dynamic capital allocation
- **Jane Street**: High-frequency trading with statistical models

### Risk Management Principles Applied
- **2% Risk Per Trade**: Standard among professional traders
- **Kelly Criterion**: Mathematical optimal position sizing
- **Half-Kelly**: Conservative 50% of full Kelly for safety
- **Sharpe Ratio Targeting**: Aim for 1.5+ (industry standard)
- **Max Drawdown Control**: Stop at daily limits

### ML Architecture Research
- **XGBoost + LSTM Hybrid**: 98%+ accuracy in 2024 studies
- **Feature Selection**: Reduces noise and overfitting
- **Sequence Learning**: LSTM captures temporal patterns
- **Walk-Forward Testing**: Realistic performance evaluation

## User Preferences

- **No Frontend Needed**: Pure API backend focus
- **High Accuracy Target**: 70%+ directional accuracy minimum
- **Industry-Grade**: Professional quant firm standards
- **Demo Mode**: Test without risk before going live
- **Comprehensive Documentation**: Technical + non-technical guides

## Configuration

Trading mode: **Demo** (paper trading with $100K virtual)  
Primary symbol: **ULTRACEMCO**  
Risk per trade: **2% of capital**  
Confidence threshold: **72%** (only trade high-quality signals)  
Daily loss limit: **$5,000** (automatic stop)  

## API Access

- **Base URL**: http://localhost:5000
- **Documentation**: http://localhost:5000/docs
- **Health Check**: http://localhost:5000/system/health

## Development Guidelines

1. **Always test in demo mode first**
2. **Review ML model performance before live trading**
3. **Monitor Sharpe ratio and drawdown closely**
4. **Use confidence filtering to avoid low-quality trades**
5. **Keep position sizes small initially**
6. **Log everything for debugging and analysis**

## Performance Targets

- **Directional Accuracy**: 70%+
- **Sharpe Ratio**: 1.5+
- **Maximum Drawdown**: <15%
- **Win Rate**: 55%+
- **Profit Factor**: 1.8+

## Safety Features

✅ Demo mode with virtual capital  
✅ Emergency kill switch endpoint  
✅ Daily and per-trade loss limits  
✅ Confidence-based trade filtering  
✅ Position size caps  
✅ Comprehensive logging and audit trail  

---

**Status**: API running, ready for model training and testing  
**Last Updated**: October 25, 2025  
**Version**: 1.0.0
