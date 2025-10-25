# Quick Start Guide

## Your Trading Bot is Ready! 🚀

The API server is already **RUNNING** on port 5000. Here's how to get started in 3 simple steps:

## Step 1: Explore the API

Visit the interactive API documentation:
- **Swagger UI**: http://localhost:5000/docs
- **Health Check**: http://localhost:5000/system/health

Try this in your browser or terminal:
```bash
curl http://localhost:5000/system/status
```

You'll see the complete system status including:
- Configuration settings
- Trading session status
- Performance metrics
- Model information

## Step 2: Train Your ML Model

Run the training script to train on 194K+ candles of ULTRACEMCO data:

```bash
python train_model.py
```

**What happens during training:**
- Loads 194,561 candles (5-minute data from 2015-2025)
- Calculates 30+ technical indicators
- Trains XGBoost for feature selection
- Trains LSTM on selected features
- Saves trained models to `trained_models/`
- Shows accuracy metrics

**Training time:** 10-20 minutes

**Expected results:**
- Directional Accuracy: 70%+
- Saved models ready for trading

## Step 3: Start Demo Trading

After training, test predictions with the API:

```bash
# Get a prediction
curl -X POST http://localhost:5000/trading/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "ULTRACEMCO",
    "open": 11500.0,
    "high": 11550.0,
    "low": 11480.0,
    "close": 11520.0,
    "volume": 125000
  }'
```

Response:
```json
{
  "symbol": "ULTRACEMCO",
  "prediction": "UP",
  "confidence": 0.7845,
  "should_trade": true,
  "timestamp": "2025-10-25T10:30:00"
}
```

## What You Have

### 18 REST API Endpoints

**System Endpoints:**
- `GET /system/health` - Health check
- `GET /system/status` - Complete system status
- `GET /system/config` - View configuration
- `POST /system/config` - Update configuration
- `GET /system/metrics` - Trading metrics

**Trading Endpoints:**
- `POST /trading/start` - Start trading session
- `POST /trading/stop` - Stop trading
- `GET /trading/status` - Session status
- `POST /trading/execute` - Execute trade
- `POST /trading/predict` - Get ML prediction
- `GET /trading/positions` - Active positions
- `GET /trading/history` - Trade history

**Admin Endpoints:**
- `POST /admin/killswitch` - Emergency stop
- `POST /admin/reset` - Reset system
- `GET /admin/logs` - View logs
- `DELETE /admin/positions/{symbol}` - Close position
- `GET /admin/system/diagnostics` - Diagnostics

### Key Features Included

✅ **Hybrid XGBoost-LSTM ML Model**
- 70%+ accuracy target
- 30+ technical indicators
- Intelligent feature selection

✅ **Professional Risk Management**
- Kelly Criterion position sizing
- 1-2% risk per trade
- Dynamic lot calculation
- Multi-layer risk controls

✅ **Demo Mode (Paper Trading)**
- $100,000 virtual capital
- Zero risk testing
- Full P&L tracking
- Easy live/demo toggle

✅ **Comprehensive Backtesting**
- Walk-forward analysis
- Sharpe ratio calculation
- Win rate & profit factor
- Maximum drawdown tracking

### Data Available

You have **4 stocks** with full historical data:
1. **ULTRACEMCO** - 194,561 candles (primary trading symbol)
2. **TVSMOTOR** - Full 5-minute data
3. **UNITDSPR** - Full 5-minute data
4. **ZYDUSLIFE** - Full 5-minute data

All data is in `data_files/` directory.

## Configuration

Edit `configs/config.yml` to customize:

```yaml
mode: demo                      # 'demo' or 'live'
symbols: [ULTRACEMCO]           # Stocks to trade
confidence_threshold: 0.72      # Min ML confidence (72%)
risk_per_trade_percent: 2.0     # Risk 2% per trade
demo_initial_capital: 100000.0  # $100K demo money
```

## Next Steps

### Immediate (Next 30 minutes)
1. ✅ API is running - explore `/docs`
2. ⏳ Train the model - run `python train_model.py`
3. 🧪 Test predictions - try the `/trading/predict` endpoint

### Short Term (Today/This Week)
4. 📊 Review training metrics - check accuracy
5. 🎮 Start demo trading - virtual $100K account
6. 📈 Monitor performance - track P&L and win rate
7. 🔧 Optimize settings - adjust confidence threshold

### Medium Term (When Ready)
8. 🔗 Connect broker API - Zerodha/AngelOne
9. 💰 Live trading - start with small positions
10. 📉 Risk management - monitor Sharpe ratio & drawdown

## Common Questions

**Q: How accurate is the model?**
A: Target is 70%+ directional accuracy. You'll see actual results after training.

**Q: Is demo mode safe?**
A: Yes! Demo mode uses virtual money. No real trades executed.

**Q: What is Kelly Criterion?**
A: Mathematical formula for optimal position sizing used by top hedge funds.

**Q: How much risk per trade?**
A: Default is 2% of capital (Half-Kelly conservative approach).

**Q: Can I trade multiple stocks?**
A: Yes! You have data for 4 stocks. Train models for each and add to config.

## Safety Tips

⚠️ **Before Going Live:**
- Always test in demo mode first
- Start with small position sizes
- Monitor drawdown closely
- Never risk more than you can afford to lose

🛡️ **Built-in Safety Features:**
- Daily loss limits
- Per-trade loss caps
- Confidence filtering (only high-quality signals)
- Emergency kill switch endpoint
- Comprehensive logging

## Need Help?

Check the full documentation:
- **README.md** - Complete technical guide
- **replit.md** - Project overview and architecture
- **API Docs** - http://localhost:5000/docs

---

**Ready to build your trading fortune with AI and mathematics! 📈🤖**
