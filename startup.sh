#!/bin/sh
apt update
apt-get install -y libgomp1
gunicorn -k uvicorn.workers.UvicornWorker main:app
streamlit run stream_lit.py --server.baseUrlPath=https://scoringmodeloc.azurewebsites.net/
