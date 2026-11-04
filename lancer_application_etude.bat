
@echo off
setlocal

title GeoPredict - Application de geotechnique

REM Se placer dans le dossier du projet
cd /d "%~dp0"

REM Verifier la presence du fichier Python
if not exist "App_Etude.py" (
    echo ERREUR : App_Etude.py est introuvable.
    pause
    exit /b 1
)

REM Verifier la presence de Python
where python >nul 2>nul
if errorlevel 1 (
    echo ERREUR : Python doit etre installe sur cet ordinateur.
    echo Telechargez-le depuis https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Verifier la presence du fichier requirements.txt
if not exist "requirements.txt" (
    echo ERREUR : requirements.txt est introuvable.
    pause
    exit /b 1
)

REM Installer les dependances
echo Installation ou verification des dependances...
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo ERREUR lors de l'installation des dependances.
    pause
    exit /b 1
)

REM Lancer GeoPredict
echo Demarrage de GeoPredict...
python -m streamlit run "App_Etude.py"

pause
endlocal
