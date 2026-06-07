@echo off
cd /d "%~dp0"

set VENV_DIR=.venv

if not exist "%VENV_DIR%" (
    echo Creant entorn virtual...
    python -m venv "%VENV_DIR%"
)

call "%VENV_DIR%\Scripts\activate.bat"

echo Instal·lant dependències...
pip install -r requirements.txt --quiet

echo.
echo === VO-evolution Dashboard ===
echo Obre http://127.0.0.1:8050 al teu navegador
echo Prem Ctrl+C per aturar
echo.
python app.py

echo.
echo Dashboard aturat.
pause
