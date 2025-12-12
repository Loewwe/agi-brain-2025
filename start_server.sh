#!/bin/bash
cd /root/Alien.agi/agi_brain
export EXCHANGE_API_KEY='8J7Y2eMoT2lwPjvcsk59pjEdZFREWmC1Q4yUs36SB0dB7Mq2cv6y66pPKZxRM4lf'
export EXCHANGE_SECRET='iCNWI7eF3AZTjFioql5rgnrZIFDip29mQ2sJfUJ9uViMOcMrYftbywkycEasFePt'
pkill -f uvicorn
nohup .venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
echo "Server started with PID $!"
