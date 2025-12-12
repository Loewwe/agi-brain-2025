#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/..
uvicorn agi_brain.src.api.main:app --host 0.0.0.0 --port 8000 --reload
