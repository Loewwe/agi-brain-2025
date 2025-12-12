#!/bin/bash
export BINANCE_API_KEY=8J7Y2eMoT2lwPjvcsk59pjEdZFREWmC1Q4yUs36SB0dB7Mq2cv6y66pPKZxRM4lf
export BINANCE_SECRET_KEY=iCNWI7eF3AZTjFioql5rgnrZIFDip29mQ2sJfUJ9uViMOcMrYftbywkycEasFePt
export SSL_CERT_FILE=/root/trade_stage6_venv/lib/python3.12/site-packages/certifi/cacert.pem

cd "/root/Trade (stage6)"

# Start Stage 6
echo "Starting Stage 6..."
nohup /root/trade_stage6_venv/bin/python3 agi_trader_stage6/main.py > agi_trader.log 2>&1 &

# Start Shadow Trader (V2)
echo "Starting Shadow Trader..."
nohup /root/trade_stage6_venv/bin/python3 agi_trader_v2/shadow_trader.py > shadow_trader.log 2>&1 &

echo "Both bots started."
