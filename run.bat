@echo off
:: EcoSight 2.0 啟動腳本
chcp 65001 > nul

set PY=C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe

echo 啟動 EcoSight 2.0...
echo 網址：http://localhost:8501
echo.

"%PY%" -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false

pause
