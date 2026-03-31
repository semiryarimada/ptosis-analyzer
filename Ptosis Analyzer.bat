@echo off
chcp 65001 >nul
title Ptosis Eyelid Analyzer

:: Kısa path kullan (boşluk sorununu önler)
for %%A in ("%~dp0.") do set "APP_DIR=%%~fA\"
for %%A in ("%~dp0.") do set "APP_SHORT=%%~sA\"

set "PYDIR=%APP_DIR%python_embedded"
set "PYSHORT=%APP_SHORT%python_embedded"
set "PYEXE=%PYDIR%\python.exe"
set "SETUP_DONE=%PYDIR%\.setup_done"
set "PORT=8502"

echo.
echo  Ptosis Eyelid Analyzer
echo  ========================
echo.

if not exist "%SETUP_DONE%" (
    call :SETUP
    if errorlevel 1 (
        echo.
        echo HATA: Kurulum basarisiz oldu.
        pause
        exit /b 1
    )
)

echo  Baslatiliyor, tarayici ~10 sn icerisinde acilacak...
echo  Bu pencereyi kapatirsan uygulama kapanir.
echo.

start /b cmd /c "timeout /t 10 /nobreak >nul && start http://localhost:%PORT%"

"%PYEXE%" -m streamlit run "%APP_DIR%main.py" ^
    --server.port %PORT% ^
    --server.headless true ^
    --browser.serverAddress localhost ^
    --server.enableCORS false ^
    --server.enableXsrfProtection false ^
    --browser.gatherUsageStats false ^
    --global.developmentMode false

echo.
echo Uygulama kapandi.
pause
goto :EOF

:SETUP
echo  ==============================================
echo   Ilk Kurulum (5-10 dakika surebilir)
echo   Bu pencereyi KAPATMAYIN!
echo  ==============================================
echo.

echo [1/4] Python indiriliyor...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; (New-Object Net.WebClient).DownloadFile('https://www.python.org/ftp/python/3.12.9/python-3.12.9-embed-amd64.zip', '%APP_SHORT%py_embed.zip'); Write-Host 'Tamam.'"
if errorlevel 1 ( echo HATA: Python indirilemedi. & pause & exit /b 1 )

echo [2/4] Python ayiklaniyor...
if not exist "%PYDIR%" mkdir "%PYDIR%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%APP_SHORT%py_embed.zip' -DestinationPath '%PYSHORT%' -Force; Write-Host 'Tamam.'"
if errorlevel 1 ( echo HATA: Ayiklama basarisiz. & pause & exit /b 1 )
del "%APP_DIR%py_embed.zip" >nul 2>&1

powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-ChildItem '%PYSHORT%\python*._pth' | ForEach-Object { $c = Get-Content $_.FullName; $c = $c -replace '#import site','import site'; if ($c -notcontains 'Lib\site-packages') { $c += 'Lib\site-packages' }; Set-Content $_.FullName $c }"

echo [3/4] pip yukleniyor...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; (New-Object Net.WebClient).DownloadFile('https://bootstrap.pypa.io/get-pip.py', '%PYSHORT%\get-pip.py')"
"%PYEXE%" "%PYDIR%\get-pip.py" --no-warn-script-location -q
if errorlevel 1 ( echo HATA: pip kurulamadi. & pause & exit /b 1 )
del "%PYDIR%\get-pip.py" >nul 2>&1

echo [4/4] Kutuphaneler yukleniyor (5-10 dk)...
"%PYEXE%" -m pip install streamlit mediapipe opencv-python-headless numpy scipy pillow matplotlib reportlab streamlit-image-coordinates --no-warn-script-location -q
if errorlevel 1 ( echo HATA: Paket kurulumu basarisiz. & pause & exit /b 1 )

echo done > "%SETUP_DONE%"
echo.
echo  Kurulum tamamlandi!
echo.
exit /b 0
