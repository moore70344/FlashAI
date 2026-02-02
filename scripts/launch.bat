@echo off
REM FlashAI Portable Launcher for Windows
REM
REM This script launches FlashAI from a flash drive, automatically
REM detecting the installation location and setting up the environment.

setlocal enabledelayedexpansion

REM Determine script location (flash drive root)
set "SCRIPT_DIR=%~dp0"
set "FLASHAI_ROOT=%SCRIPT_DIR%.."
pushd "%FLASHAI_ROOT%"
set "FLASHAI_ROOT=%CD%"
popd

echo ================================================================
echo                      FlashAI Launcher
echo         Portable AI System with Energy-Based Reasoning
echo ================================================================
echo.
echo Installation: %FLASHAI_ROOT%
echo.

REM Set environment variables
set "FLASHAI_BASE=%FLASHAI_ROOT%"
set "FLASHAI_PORTABLE=1"
set "FLASHAI_DATA=%FLASHAI_ROOT%\data"
set "FLASHAI_CONFIG=%FLASHAI_ROOT%\config"

REM Check for portable Python
set "PORTABLE_PYTHON=%FLASHAI_ROOT%\python\python.exe"
if exist "%PORTABLE_PYTHON%" (
    set "PYTHON=%PORTABLE_PYTHON%"
    echo Using portable Python: %PYTHON%
) else (
    REM Use system Python
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        set "PYTHON=python"
    ) else (
        where python3 >nul 2>nul
        if %errorlevel% equ 0 (
            set "PYTHON=python3"
        ) else (
            echo Error: Python not found. Please install Python 3.9+ or bundle portable Python.
            pause
            exit /b 1
        )
    )
    echo Using system Python: !PYTHON!
)

REM Check Python version
for /f "tokens=*" %%i in ('!PYTHON! -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set "PY_VERSION=%%i"
echo Python version: %PY_VERSION%

REM Check if FlashAI is installed
!PYTHON! -c "import flashai" >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo FlashAI not installed. Installing...
    pushd "%FLASHAI_ROOT%"
    !PYTHON! -m pip install -e . --quiet
    popd
    echo Installation complete.
)

echo.
echo Starting FlashAI...
echo.

REM Parse arguments
set "ACTION=%1"
if "%ACTION%"=="" set "ACTION=serve"

REM Shift arguments
shift

if /i "%ACTION%"=="serve" (
    !PYTHON! -m flashai.cli serve --base-path "%FLASHAI_ROOT%" %*
) else if /i "%ACTION%"=="reason" (
    !PYTHON! -m flashai.cli reason --base-path "%FLASHAI_ROOT%" %*
) else if /i "%ACTION%"=="reset" (
    !PYTHON! -m flashai.cli reset --base-path "%FLASHAI_ROOT%" %*
) else if /i "%ACTION%"=="restore" (
    !PYTHON! -m flashai.cli restore --base-path "%FLASHAI_ROOT%" %*
) else if /i "%ACTION%"=="status" (
    !PYTHON! -m flashai.cli status --base-path "%FLASHAI_ROOT%" %*
) else if /i "%ACTION%"=="init" (
    !PYTHON! -m flashai.cli init --base-path "%FLASHAI_ROOT%" %*
) else if /i "%ACTION%"=="help" (
    echo Usage: launch.bat [command] [options]
    echo.
    echo Commands:
    echo   serve     Start the FlashAI API server (default)
    echo   reason    Perform reasoning on a query
    echo   reset     Save and reset to initial state
    echo   restore   Restore from an export file
    echo   status    Show system status
    echo   init      Initialize FlashAI
    echo   help      Show this help message
    echo.
    echo Examples:
    echo   launch.bat serve --port 8080
    echo   launch.bat reason "What is 2+2?"
    echo   launch.bat reset --name backup
) else if /i "%ACTION%"=="--help" (
    goto :help
) else if /i "%ACTION%"=="-h" (
    goto :help
) else (
    !PYTHON! -m flashai.cli %ACTION% --base-path "%FLASHAI_ROOT%" %*
)

goto :eof

:help
echo Usage: launch.bat [command] [options]
echo.
echo Commands:
echo   serve     Start the FlashAI API server (default)
echo   reason    Perform reasoning on a query
echo   reset     Save and reset to initial state
echo   restore   Restore from an export file
echo   status    Show system status
echo   init      Initialize FlashAI
echo   help      Show this help message
goto :eof

endlocal
