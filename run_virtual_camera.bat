@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create .venv and install requirements-virtual.txt first. See README.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" virtual_camera.py %*
pause
